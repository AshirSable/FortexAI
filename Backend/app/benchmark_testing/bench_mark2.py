import logging
import time
from pathlib import Path
import polars as pl

from app.pipeline.semantic_search import SemanticSearchPipeline
from app.type_store import PhaseInput, Verdict

logger = logging.getLogger(__name__)


def run_poisoning_experiment(
    test_prompts: list[str],
    test_labels: list[int],  # 0 = benign, 1 = attack
    poison_prompts: list[str],  # Benign prompts to intentionally insert as "attack"
    out_path: Path,
    similarity_threshold: float = 0.85,
) -> pl.DataFrame:
    """
    Stress-tests the SemanticSearchPipeline against index poisoning.

    1. Evaluates baseline performance on a clean test set.
    2. Inject `poison_prompts` into the vector store marked as "attack".
    3. Re-evaluates test set to measure False Positive Rate (FPR) shift.
    """
    # 1. Instantiate clean Semantic Search Phase
    search_phase = SemanticSearchPipeline()

    def evaluate_search_stage(prompts: list[str], labels: list[int], state_label: str):
        """Helper to run prompts through Semantic Search and compute metrics."""
        results = []
        for idx, (prompt, ground_truth) in enumerate(zip(prompts, labels)):
            start_time = time.perf_counter()
            res = search_phase.verdict(PhaseInput(text=prompt))
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if res.is_err():
                verdict_str = "error"
                conf = 0.0
            else:
                success = res.unwrap()
                verdict_str = success.verdict.name
                conf = success.confidence

            # 0 = benign, 1 = attack
            is_fp = (ground_truth == 0) and (verdict_str == "attack")
            is_tp = (ground_truth == 1) and (verdict_str == "attack")
            is_fn = (ground_truth == 1) and (
                verdict_str == "benign" or verdict_str == "undetermined"
            )
            is_tn = (ground_truth == 0) and (
                verdict_str == "benign" or verdict_str == "undetermined"
            )

            results.append(
                {
                    "eval_state": state_label,
                    "prompt_idx": idx,
                    "ground_truth": ground_truth,
                    "predicted_verdict": verdict_str,
                    "confidence": conf,
                    "latency_ms": elapsed_ms,
                    "is_fp": is_fp,
                    "is_tp": is_tp,
                    "is_fn": is_fn,
                    "is_tn": is_tn,
                }
            )
        return results

    # Step A: Measure Baseline Performance
    print("--- Running Baseline Evaluation (Clean Index) ---")
    baseline_rows = evaluate_search_stage(
        test_prompts, test_labels, state_label="baseline"
    )

    # Step B: Inject Poisoned Entries
    print(
        f"\n--- Injecting {len(poison_prompts)} Poisoned Entries into Vector Store ---"
    )
    poison_count = 0
    for p_prompt in poison_prompts:
        try:
            # Force-confirm benign prompts as "attack"
            search_phase.confirm(p_prompt, label="attack")
            poison_count += 1
        except Exception as e:
            logger.warning("Failed to inject poison prompt: %s", e)

    print(f"Successfully poisoned {poison_count} entries.\n")

    # Step C: Re-evaluate Performance on Post-Poisoning Index
    print("--- Running Post-Poisoning Evaluation ---")
    poisoned_rows = evaluate_search_stage(
        test_prompts, test_labels, state_label="poisoned"
    )

    # Combine & Compute Summary Statistics
    all_rows = baseline_rows + poisoned_rows
    df = pl.DataFrame(all_rows)

    # Compute aggregate metrics
    metrics = (
        df.group_by("eval_state")
        .agg(
            total_benign=pl.col("ground_truth").eq(0).sum(),
            total_attacks=pl.col("ground_truth").eq(1).sum(),
            false_positives=pl.col("is_fp").sum(),
            true_positives=pl.col("is_tp").sum(),
            false_negatives=pl.col("is_fn").sum(),
            true_negatives=pl.col("is_tn").sum(),
        )
        .with_columns(
            fpr=(pl.col("false_positives") / pl.col("total_benign") * 100).round(2),
            tpr_recall=(pl.col("true_positives") / pl.col("total_attacks") * 100).round(
                2
            ),
        )
    )

    print("\n" + "=" * 50)
    print("POISONING EXPERIMENT RESULTS")
    print("=" * 50)
    print(metrics)

    # Save granular result log
    df.write_csv(out_path)
    print(f"\nDetailed prompt-level metrics exported to {out_path}")

    # Calculate exact delta to quote in paper
    baseline_fpr = metrics.filter(pl.col("eval_state") == "baseline")["fpr"][0]
    poisoned_fpr = metrics.filter(pl.col("eval_state") == "poisoned")["fpr"][0]
    delta_fpr = round(poisoned_fpr - baseline_fpr, 2)

    print(f"\nSummary for Reviewers:")
    print(f"-> Baseline FPR: {baseline_fpr}%")
    print(f"-> Post-Poisoning FPR ({poison_count} injected entries): {poisoned_fpr}%")
    print(f"-> Net FPR Shift: +{delta_fpr}%")

    return df
