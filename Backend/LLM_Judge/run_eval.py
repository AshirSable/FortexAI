"""Parallel, resumable evaluation driver over a frozen sample parquet
(make_sample.py output). Backend-agnostic: it calls
guardrails.evaluate_with_guardrails(), which honours LLM_JUDGE_BACKEND
(ollama | groq) and LLM_JUDGE_SELF_CONSISTENCY.

Designed for a rate-limited cloud backend and for runs that may be interrupted:
  - writes each result to the CSV immediately (resume skips rows already there)
  - a few worker threads (Groq handles concurrency; the free tier's real cap is
    tokens/min, so 4-8 workers is plenty)
  - ret/backoff on 429/5xx lives in judge._chat_via_groq

    LLM_JUDGE_BACKEND=groq GROQ_API_KEY=... \
      python run_eval.py eval_sample.parquet --out results.csv --workers 6

Re-run the same command after an interruption to continue. `python run_eval.py
--metrics-only results.csv` recomputes metrics from an existing CSV.
"""
import argparse
import collections
import csv
import math
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

LABEL_MAP = {1: "malicious", 0: "benign"}
FIELDS = ["idx", "prompt", "expected_verdict", "model_verdict",
          "model_confidence", "model_reason", "correct"]


class RateLimiter:
    """Blocks so that no more than `rpm` acquisitions happen per rolling minute.
    rpm <= 0 disables it. Groq free tier is ~8k tokens/min ~= 6 calls/min."""

    def __init__(self, rpm: float):
        self.rpm = rpm
        self._times: collections.deque[float] = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60:
                    self._times.popleft()
                if len(self._times) < self.rpm:
                    self._times.append(now)
                    return
                sleep_for = 60 - (now - self._times[0]) + 0.05
            time.sleep(max(sleep_for, 0.05))


def load_done(out_csv: Path) -> set[int]:
    """idxs that already have a real (non-ERROR) result - those are skipped on
    resume. ERROR rows are left to be retried."""
    if not out_csv.exists():
        return set()
    try:
        df = pd.read_csv(out_csv)
        return set(df.loc[df["model_verdict"] != "ERROR", "idx"].astype(int).tolist())
    except Exception:
        return set()


def run(sample_parquet: Path, out_csv: Path, workers: int, limit: int | None, rpm: float) -> None:
    import guardrails
    from judge import JudgeUnavailable

    self_consistency = guardrails._default_self_consistency()
    df = pd.read_parquet(sample_parquet).reset_index(drop=True)
    if limit:
        df = df.iloc[:limit]

    done = load_done(out_csv)
    todo = [(i, df.at[i, "text"], int(df.at[i, "label"])) for i in df.index if i not in done]
    print(f"{len(df)} rows in sample | {len(done)} already done | {len(todo)} to evaluate")
    print(f"backend={os.getenv('LLM_JUDGE_BACKEND', 'ollama')} "
          f"model={os.getenv('GROQ_MODEL', '') or 'ollama default'} "
          f"self_consistency={self_consistency} workers={workers} rpm={rpm or 'unlimited'}\n")
    if not todo:
        return

    limiter = RateLimiter(rpm)
    lock = threading.Lock()
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    f = out_csv.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()
        f.flush()

    def work(item):
        i, text, label = item
        expected = LABEL_MAP[label]
        try:
            limiter.acquire()
            v = guardrails.evaluate_with_guardrails(
                text, self_consistency=self_consistency, strict=True
            )
            return {"idx": i, "prompt": text, "expected_verdict": expected,
                    "model_verdict": v.verdict.value, "model_confidence": round(v.confidence, 4),
                    "model_reason": v.reason, "correct": v.verdict.value == expected}
        except JudgeUnavailable as e:
            return {"idx": i, "prompt": text, "expected_verdict": expected,
                    "model_verdict": "ERROR", "model_confidence": 0.0,
                    "model_reason": f"unavailable: {e}", "correct": False}
        except Exception as e:
            traceback.print_exc()
            return {"idx": i, "prompt": text, "expected_verdict": expected,
                    "model_verdict": "ERROR", "model_confidence": 0.0,
                    "model_reason": f"{type(e).__name__}: {e}", "correct": False}

    start = time.time()
    n = hits = errs = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(work, it) for it in todo]):
            r = fut.result()
            with lock:
                writer.writerow(r)
                f.flush()
            n += 1
            if r["model_verdict"] == "ERROR":
                errs += 1
            else:
                hits += int(r["correct"])
            if n % 25 == 0 or n == len(todo):
                el = time.time() - start
                rate = n / el * 60
                eta = (len(todo) - n) / (n / el) / 60 if n else 0
                ok = n - errs
                acc = hits / ok if ok else 0.0
                print(f"[{n}/{len(todo)}] {rate:5.1f}/min  ETA {eta:6.1f} min  "
                      f"running acc {acc:.3f}  errors {errs}")
    f.close()
    print(f"\nsession done: {n} attempted, {errs} still errored (re-run to retry them)")


def metrics(out_csv: Path) -> dict:
    res = pd.read_csv(out_csv).drop_duplicates(subset="idx", keep="last")
    errs = int((res.model_verdict == "ERROR").sum())
    res = res[res.model_verdict != "ERROR"]
    n = len(res)

    def c(exp, pred):
        return int(((res.expected_verdict == exp) & (res.model_verdict == pred)).sum())

    tp, fn, mu = c("malicious", "malicious"), c("malicious", "benign"), c("malicious", "uncertain")
    fp, tn, bu = c("benign", "malicious"), c("benign", "benign"), c("benign", "uncertain")
    n_mal, n_ben = tp + fn + mu, fp + tn + bu

    acc = int(res.correct.sum()) / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_mal if n_mal else 0.0
    rec_esc = (tp + mu) / n_mal if n_mal else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tnr = tn / n_ben if n_ben else 0.0
    bal_acc = (rec + tnr) / 2

    z = 1.96
    den = 1 + z ** 2 / n if n else 1
    centre = (acc + z ** 2 / (2 * n)) / den if n else 0.0
    half = (z * math.sqrt(acc * (1 - acc) / n + z ** 2 / (4 * n ** 2)) / den) if n else 0.0

    m = {
        "n_evaluated": n, "n_errors": errs, "n_malicious": n_mal, "n_benign": n_ben,
        "accuracy": acc, "accuracy_95ci": [round(centre - half, 4), round(centre + half, 4)],
        "balanced_accuracy": bal_acc, "precision_malicious": prec,
        "recall_malicious": rec, "recall_malicious_with_escalation": rec_esc,
        "f1_malicious": f1, "specificity_benign": tnr,
        "confusion_matrix": {
            "malicious_expected": {"malicious": tp, "benign": fn, "uncertain": mu},
            "benign_expected": {"malicious": fp, "benign": tn, "uncertain": bu},
        },
    }
    return m


def print_metrics(m: dict) -> None:
    print("\n===== METRICS (positive class = malicious) =====")
    print(f"  evaluated        {m['n_evaluated']}  ({m['n_malicious']} malicious / {m['n_benign']} benign)"
          + (f"   [{m['n_errors']} errored rows excluded]" if m["n_errors"] else ""))
    ci = m["accuracy_95ci"]
    print(f"  accuracy         {m['accuracy']:.4f}   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  balanced acc     {m['balanced_accuracy']:.4f}")
    print(f"  precision (mal)  {m['precision_malicious']:.4f}")
    print(f"  recall (mal)     {m['recall_malicious']:.4f}   (+escalation {m['recall_malicious_with_escalation']:.4f})")
    print(f"  F1 (mal)         {m['f1_malicious']:.4f}")
    print(f"  specificity      {m['specificity_benign']:.4f}")
    cm = m["confusion_matrix"]
    me, be = cm["malicious_expected"], cm["benign_expected"]
    print("\n  confusion            pred:malicious  pred:benign  pred:uncertain")
    print(f"    exp malicious  {me['malicious']:>14}  {me['benign']:>11}  {me['uncertain']:>14}")
    print(f"    exp benign     {be['malicious']:>14}  {be['benign']:>11}  {be['uncertain']:>14}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", type=Path, help="sample parquet to evaluate, or the results CSV with --metrics-only")
    p.add_argument("--out", type=Path, default=Path("results.csv"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--rpm", type=float, default=0,
                   help="cap requests/minute (Groq free tier: use ~5). 0 = unlimited")
    p.add_argument("--limit", type=int, default=None, help="only the first N rows of the sample (smoke test)")
    p.add_argument("--metrics-only", action="store_true", help="recompute metrics from an existing results CSV")
    args = p.parse_args()

    if args.metrics_only:
        m = metrics(args.target)
        print_metrics(m)
        return

    run(args.target, args.out, args.workers, args.limit, args.rpm)
    m = metrics(args.out)
    print_metrics(m)
    import json
    mp = args.out.with_suffix(".metrics.json")
    mp.write_text(json.dumps(m, indent=2))
    print(f"\nresults: {args.out}\nmetrics: {mp}")


if __name__ == "__main__":
    main()
