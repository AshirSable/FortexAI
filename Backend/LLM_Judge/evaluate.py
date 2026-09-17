"""Evaluation harness: run the judge against data/labeled_test_set.json and
report accuracy, per-label breakdown, and false-positive/false-negative counts.

Calls guardrails.evaluate_with_guardrails() - the hardened, production-recommended
path (input isolation + hidden base64/ROT13 payload decoding + self-consistency),
not the raw judge alone - since that's what should actually be measured/trusted.

    python evaluate.py
"""

import csv
import json
from pathlib import Path

import polars as pl
from LLM_Judge import guardrails
from ml_factory import DATA_PROCESSED_DIR, DATA_RAW_DIR
from ml_factory.datasets.sampler import SplitSampler
from ml_factory.utils import give_id_to_data

LABELED_SET_PATH = Path(__file__).parent / "data" / "labeled_test_set.json"
OUTPUT_CSV_PATH = Path(__file__).parent / "evaluation_results.csv"

CSV_FIELDS = [
    "prompt",
    "category",
    "expected_verdict",
    "model_verdict",
    "model_confidence",
    "model_reason",
    "correct",
]


def load_labeled_set(path: Path = LABELED_SET_PATH) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def load_test_dataset(head: int | None = 100) -> list[dict]:

    data_file = DATA_RAW_DIR / "final_data.parquet"
    split_file = DATA_PROCESSED_DIR / "split_data_seed_3123_train_70_test_15.pt"
    df = pl.read_parquet(data_file)
    df = give_id_to_data(df)
    if split_file.exists():
        sampler = SplitSampler(
            train_split=0.7, test_split=0.15, validation_split=0.15, seed=444
        )
        sampler.build_split(df["id"], df["label"])
        sampler.save_split(split_file)
    else:
        sampler = SplitSampler.load_file(path=split_file)

    df_filtered = df.filter(pl.col("id").is_in(sampler.get_split("test")))
    df_filtered = df_filtered.with_columns(
        pl.when(pl.col("label") == 1)
        .then(pl.lit("malicious"))
        .otherwise(pl.lit("benign"))
        .alias("label")
    )
    data = df_filtered.rename({"text": "input", "label": "expected_verdict"}).select(
        "input", "expected_verdict"
    )
    if head is not None:
        data = data.head(head)

    return data.to_dicts()


def run_evaluation(cases: list[dict]) -> list[dict]:
    results = []
    for i, case in enumerate(cases, 1):
        verdict = guardrails.evaluate_with_guardrails(
            case["input"], context=case.get("context")
        )
        correct = verdict.verdict.value == case["expected_verdict"]
        results.append(
            {
                "prompt": case["input"],
                "category": case.get("category", ""),
                "expected_verdict": case["expected_verdict"],
                "model_verdict": verdict.verdict.value,
                "model_confidence": verdict.confidence,
                "model_reason": verdict.reason,
                "correct": correct,
            }
        )
        status = "OK" if correct else "MISMATCH"
        print(
            f"[{i}/{len(cases)}] expected={case['expected_verdict']:10s} "
            f"got={verdict.verdict.value:10s} {status} text got: {case['input']:<50}"
        )
    return results


def write_csv(results: list[dict], path: Path = OUTPUT_CSV_PATH) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]) -> None:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"\nOverall accuracy: {correct}/{total} ({correct / total:.1%})")

    print("\nPer-expected-label breakdown:")
    for label in ("malicious", "benign", "uncertain"):
        subset = [r for r in results if r["expected_verdict"] == label]
        if not subset:
            continue
        subset_correct = sum(r["correct"] for r in subset)
        print(
            f"  {label:10s}: {subset_correct}/{len(subset)} ({subset_correct / len(subset):.1%})"
        )

    malicious_cases = [r for r in results if r["expected_verdict"] == "malicious"]
    false_negatives = [r for r in malicious_cases if r["model_verdict"] == "benign"]
    if malicious_cases:
        print(
            f"\nMalicious -> benign false negatives (the dangerous kind): "
            f"{len(false_negatives)}/{len(malicious_cases)}"
        )
        for r in false_negatives:
            print(f"    [{r['category']}] {r['prompt'][:80]!r}")

    benign_cases = [r for r in results if r["expected_verdict"] == "benign"]
    false_positives = [r for r in benign_cases if r["model_verdict"] == "malicious"]
    if benign_cases:
        print(
            f"Benign -> malicious false positives: {len(false_positives)}/{len(benign_cases)}"
        )
        for r in false_positives:
            print(f"    [{r['category']}] {r['prompt'][:80]!r}")


def main() -> None:
    cases = load_test_dataset(20)
    results = run_evaluation(cases)
    write_csv(results)
    print_summary(results)
    print(f"\nWrote {len(results)} rows to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
