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
    print(f"\nWrote {len(results)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
