"""Evaluation harness for external parquet-format datasets (text, label columns,
label 1=malicious/0=benign), as opposed to evaluate.py which runs the small
hand-curated data/labeled_test_set.json.

Meant for larger, harder external sets (e.g. paraphrase/synonym-obfuscated
jailbreak datasets) where running every row isn't practical against a local
model - dedupes exact-duplicate text, then takes a class-balanced random
sample of a given size and runs it through guardrails.evaluate_with_guardrails()
(the same production-recommended path evaluate.py measures).

    python evaluate_parquet.py <path_to.parquet> [--sample-size 500] [--seed 42]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

import guardrails

LABEL_MAP = {1: "malicious", 0: "benign"}

CSV_FIELDS = [
    "prompt",
    "expected_verdict",
    "model_verdict",
    "model_confidence",
    "model_reason",
    "correct",
]


def load_and_sample(path: Path, sample_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    before = len(df)
    df = df.drop_duplicates(subset="text")
    df = df[df["text"].str.strip() != ""]
    deduped = len(df)

    per_class = sample_size // 2
    parts = []
    for label in (0, 1):
        subset = df[df["label"] == label]
        n = min(per_class, len(subset))
        parts.append(subset.sample(n=n, random_state=seed))
    sampled = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Loaded {before} rows, {deduped} after deduping/dropping empties.")
    print(f"Sampled {len(sampled)} rows "
          f"({(sampled['label'] == 1).sum()} malicious / {(sampled['label'] == 0).sum()} benign).")
    return sampled


def run_evaluation(df: pd.DataFrame, output_csv: Path) -> list[dict]:
    results = []
    start = time.time()
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        for i, row in enumerate(df.itertuples(index=False), 1):
            expected = LABEL_MAP[row.label]
            verdict = guardrails.evaluate_with_guardrails(row.text)
            correct = verdict.verdict.value == expected
            result = {
                "prompt": row.text,
                "expected_verdict": expected,
                "model_verdict": verdict.verdict.value,
                "model_confidence": verdict.confidence,
                "model_reason": verdict.reason,
                "correct": correct,
            }
            results.append(result)
            writer.writerow(result)
            f.flush()
            elapsed = time.time() - start
            eta = elapsed / i * (len(df) - i)
            status = "OK" if correct else "MISMATCH"
            print(
                f"[{i}/{len(df)}] expected={expected:10s} got={verdict.verdict.value:10s} "
                f"{status}  (elapsed={elapsed:.0f}s, eta={eta:.0f}s)"
            )
    return results


def print_summary(results: list[dict]) -> None:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"\nOverall accuracy: {correct}/{total} ({correct / total:.1%})")

    print("\nPer-expected-label breakdown:")
    for label in ("malicious", "benign"):
        subset = [r for r in results if r["expected_verdict"] == label]
        if not subset:
            continue
        subset_correct = sum(r["correct"] for r in subset)
        print(f"  {label:10s}: {subset_correct}/{len(subset)} ({subset_correct / len(subset):.1%})")

    malicious_cases = [r for r in results if r["expected_verdict"] == "malicious"]
    false_negatives = [r for r in malicious_cases if r["model_verdict"] == "benign"]
    escalated = [r for r in malicious_cases if r["model_verdict"] == "uncertain"]
    if malicious_cases:
        print(
            f"\nMalicious -> benign false negatives (the dangerous kind): "
            f"{len(false_negatives)}/{len(malicious_cases)}"
        )
        print(f"Malicious -> uncertain (escalated, not silently missed): {len(escalated)}/{len(malicious_cases)}")
        for r in false_negatives[:15]:
            print(f"    {r['prompt'][:100]!r}")
        if len(false_negatives) > 15:
            print(f"    ... and {len(false_negatives) - 15} more (see CSV)")

    benign_cases = [r for r in results if r["expected_verdict"] == "benign"]
    false_positives = [r for r in benign_cases if r["model_verdict"] == "malicious"]
    if benign_cases:
        print(f"\nBenign -> malicious false positives: {len(false_positives)}/{len(benign_cases)}")
        for r in false_positives[:15]:
            print(f"    {r['prompt'][:100]!r}")
        if len(false_positives) > 15:
            print(f"    ... and {len(false_positives) - 15} more (see CSV)")


def compute_metrics(results: list[dict]) -> dict:
    """Binary metrics with `malicious` as the positive class. The judge can also
    answer `uncertain` (escalate to a human); it's reported as its own bucket
    and, for recall, also credited as "not silently missed"."""
    def n(expected, predicted):
        return sum(
            1 for r in results
            if r["expected_verdict"] == expected and r["model_verdict"] == predicted
        )

    tp = n("malicious", "malicious")
    fn = n("malicious", "benign")
    mal_uncertain = n("malicious", "uncertain")
    fp = n("benign", "malicious")
    tn = n("benign", "benign")
    ben_uncertain = n("benign", "uncertain")

    total = len(results)
    n_mal = tp + fn + mal_uncertain
    n_ben = fp + tn + ben_uncertain
    correct = sum(r["correct"] for r in results)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall_strict = tp / n_mal if n_mal else 0.0
    recall_with_escalation = (tp + mal_uncertain) / n_mal if n_mal else 0.0
    f1 = (2 * precision * recall_strict / (precision + recall_strict)
          if (precision + recall_strict) else 0.0)
    tnr = tn / n_ben if n_ben else 0.0
    balanced_acc = (recall_strict + tnr) / 2

    return {
        "n_total": total,
        "n_malicious": n_mal,
        "n_benign": n_ben,
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": balanced_acc,
        "precision_malicious": precision,
        "recall_malicious_strict": recall_strict,
        "recall_malicious_with_escalation": recall_with_escalation,
        "f1_malicious": f1,
        "specificity_benign": tnr,
        "confusion_matrix": {
            "malicious_expected": {"malicious": tp, "benign": fn, "uncertain": mal_uncertain},
            "benign_expected": {"malicious": fp, "benign": tn, "uncertain": ben_uncertain},
        },
    }


def print_metrics(m: dict) -> None:
    print("\n===== METRICS (positive class = malicious) =====")
    print(f"  Samples:            {m['n_total']}  ({m['n_malicious']} malicious / {m['n_benign']} benign)")
    print(f"  Accuracy:           {m['accuracy']:.4f}")
    print(f"  Balanced accuracy:  {m['balanced_accuracy']:.4f}")
    print(f"  Precision:          {m['precision_malicious']:.4f}")
    print(f"  Recall (strict):    {m['recall_malicious_strict']:.4f}")
    print(f"  Recall (+escalate): {m['recall_malicious_with_escalation']:.4f}")
    print(f"  F1:                 {m['f1_malicious']:.4f}")
    print(f"  Specificity:        {m['specificity_benign']:.4f}")
    cm = m["confusion_matrix"]
    print("\n  Confusion matrix          pred:malicious  pred:benign  pred:uncertain")
    me, be = cm["malicious_expected"], cm["benign_expected"]
    print(f"    expected malicious {me['malicious']:>14}  {me['benign']:>11}  {me['uncertain']:>14}")
    print(f"    expected benign    {be['malicious']:>14}  {be['benign']:>11}  {be['uncertain']:>14}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_path", type=Path)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=Path(__file__).parent / "parquet_evaluation_results.csv")
    args = parser.parse_args()

    df = load_and_sample(args.parquet_path, args.sample_size, args.seed)
    results = run_evaluation(df, args.output_csv)
    print_summary(results)

    metrics = compute_metrics(results)
    print_metrics(metrics)
    metrics_path = args.output_csv.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nWrote {len(results)} rows to {args.output_csv}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
