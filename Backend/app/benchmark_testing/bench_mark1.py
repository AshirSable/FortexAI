import time
import polars as pl
from pathlib import Path

from app.main import CONFIRM_CONFIDENCE_THRESHOLD
from app.pipeline.autoencoder import AutoEncoderPipeline
from app.pipeline.bert import EnsembleBERTPipeline
from app.pipeline.llm_judge import LLM_JudgePipeline
from app.pipeline.semantic_search import SemanticSearchPipeline
from app.type_store import Phase, PhaseInput, Verdict


def benchmark_pipeline_on_ood(
    prompts: list[str],
    labels: list[int],  # 0 = benign, 1 = attack, ground truth
    categories: list[str] | None,
    out_path: Path,
):
    """
    Runs EVERY phase on EVERY prompt unconditionally to collect complete
    per-stage benchmark metrics, while marking 'would_stop_here' and tracking
    the actual pipeline outcome in the FINAL summary row.
    """
    phases = [
        SemanticSearchPipeline(),
        AutoEncoderPipeline(),
        EnsembleBERTPipeline(),
        LLM_JudgePipeline(),
    ]

    rows = []

    for idx, (prompt, label) in enumerate(zip(prompts, labels)):
        category = categories[idx] if categories is not None else None
        current_input = PhaseInput(text=prompt)
        prior_results = {}

        # Tracks what the real pipeline decision would have been
        final_verdict = None
        final_phase = None
        final_confidence = None
        final_latency_ms = None
        pipeline_stopped = False

        cumulative_start = time.perf_counter()

        for phase in phases:
            phase_start = time.perf_counter()
            result = phase.verdict(current_input)
            phase_elapsed_ms = (time.perf_counter() - phase_start) * 1000
            cumulative_elapsed_ms = (time.perf_counter() - cumulative_start) * 1000

            if result.is_err():
                rows.append(
                    {
                        "prompt_idx": idx,
                        "label": label,
                        "category": category,
                        "phase": phase.name,
                        "verdict": "error",
                        "confidence": None,
                        "phase_latency_ms": phase_elapsed_ms,
                        "cumulative_latency_ms": cumulative_elapsed_ms,
                        "would_stop_here": True,
                        "ran_after_stop": pipeline_stopped,
                    }
                )
                if not pipeline_stopped:
                    final_verdict = "error"
                    final_phase = phase.phase.name
                    final_confidence = 0.0
                    final_latency_ms = cumulative_elapsed_ms
                    pipeline_stopped = True
                continue  # Continue to next phase instead of breaking

            success = result.unwrap()

            # Determine if real pipeline criteria would stop at this stage
            would_stop = False
            if (
                phase.phase == Phase.semantic_search
                and success.verdict == Verdict.attack
            ):
                would_stop = True
            elif phase.phase == Phase.autoencoder and success.verdict == Verdict.benign:
                would_stop = True
            elif phase.phase == Phase.ensemble_bert and success.verdict in (
                Verdict.benign,
                Verdict.attack,
            ):
                would_stop = True
            elif phase.phase == Phase.llm_judge:
                would_stop = True

            rows.append(
                {
                    "prompt_idx": idx,
                    "label": label,
                    "category": category,
                    "phase": phase.name,
                    "verdict": success.verdict.name,
                    "confidence": success.confidence,
                    "phase_latency_ms": phase_elapsed_ms,
                    "cumulative_latency_ms": cumulative_elapsed_ms,
                    "would_stop_here": would_stop,
                    "ran_after_stop": pipeline_stopped,
                }
            )

            # Record the actual cascade output at the moment of first stopping decision
            if would_stop and not pipeline_stopped:
                final_verdict = success.verdict.name
                final_phase = phase.phase.name
                final_confidence = success.confidence
                final_latency_ms = cumulative_elapsed_ms
                pipeline_stopped = True

            # Pass history down to the next stage regardless
            prior_results[phase.phase] = success
            current_input = PhaseInput(text=prompt, _prior_results=prior_results)

        # Fallback if no stage triggered stopping criteria
        if not pipeline_stopped and "success" in locals():
            final_verdict = success.verdict.name
            final_phase = phase.phase.name
            final_confidence = success.confidence
            final_latency_ms = cumulative_elapsed_ms

        # Write-back to index using the cascade's final decision
        if (
            final_phase is not None
            and final_phase != Phase.semantic_search.name
            and final_confidence is not None
            and final_confidence >= CONFIRM_CONFIDENCE_THRESHOLD
        ):
            try:
                write_label = (
                    "attack" if final_verdict == Verdict.attack.name else "benign"
                )
                phases[0].confirm(prompt, write_label)
            except Exception as e:
                print(f"semantic search write-back failed for prompt_idx={idx}: {e}")

        # Summary row representing real-world cascade outcome
        rows.append(
            {
                "prompt_idx": idx,
                "label": label,
                "category": category,
                "phase": "FINAL",
                "verdict": final_verdict,
                "confidence": final_confidence,
                "phase_latency_ms": None,
                "cumulative_latency_ms": final_latency_ms,
                "would_stop_here": True,
                "ran_after_stop": False,
            }
        )

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(prompts)}")

    df = pl.DataFrame(rows)
    df.write_csv(out_path)
    print(f"Wrote {len(df)} rows ({len(prompts)} prompts) to {out_path}")
    return df
