"""
Stage 4 of the cascade: LLM judge. This is the LAST stage - if a prompt gets
here, every earlier stage (semantic search, autoencoder, mini-BERT) looked at
it and had no opinion.

Wraps Backend/LLM_Judge/guardrails.py's evaluate_with_guardrails() - that
module's own documented production entry point (see LLM_Judge/CLAUDE.md:
"production code should always go through guardrails.py, never call
judge.evaluate_prompt directly"). It already handles: isolating the flagged
text behind a random boundary, decoding hidden base64/ROT13 payloads and
judging those too, and (by default) a two-pass self-consistency check. None
of that is reimplemented here - we only map its Verdict onto ours.

Which model backend it actually talks to (local Ollama vs. Groq) is
controlled entirely inside LLM_Judge/config.py via environment variables
(LLM_JUDGE_BACKEND, LLM_JUDGE_MODEL, GROQ_MODEL, ...) - not this file.
"""

import numpy as np

from LLM_Judge.guardrails import evaluate_with_guardrails
from LLM_Judge.schema import VerdictLabel

from app.pipeline import PipelinePhase
from app.pipeline.utils import embed_text
from app.type_store import (
    Err,
    Ok,
    Phase,
    PhaseInput,
    Result,
    SuccessForReview,
    SuccessReturn,
    Verdict,
)
from app.type_store._error import InferenceError, PhaseError

# This is the last stage, so "uncertain" can't be passed further down the
# cascade the way earlier stages pass "undetermined" along. LLM_Judge's own
# fail-safe direction is "never a silent benign" (see judge.py's _fail_safe) -
# we keep that same direction here: an uncertain final verdict fails closed
# to attack, just at whatever (typically low) confidence the judge reported,
# so it's visibly a weak call rather than a confident one.
_VERDICT_MAP = {
    VerdictLabel.MALICIOUS: Verdict.attack,
    VerdictLabel.BENIGN: Verdict.benign,
    VerdictLabel.UNCERTAIN: Verdict.attack,
}


class LLM_JudgePipeline(PipelinePhase):
    def __init__(self):
        super().__init__(phase=Phase.llm_judge)

    def verdict(self, input: PhaseInput) -> Result[SuccessReturn, PhaseError]:
        text = input.require_text()

        try:
            judge_result = evaluate_with_guardrails(text)
        except Exception as e:
            return Err(InferenceError(f"llm judge failed: {e}"))

        verdict = _VERDICT_MAP[judge_result.verdict]

        return Ok(
            SuccessReturn(
                verdict=verdict, at_phase=self.phase, confidence=judge_result.confidence
            )
        )

    def verdict_with_data(
        self, input: PhaseInput
    ) -> Result[SuccessForReview, PhaseError]:
        result = self.verdict(input)
        if result.is_err():
            return result

        text = input.require_text()
        embedding = (
            input.embedding
            if input.embedding is not None
            else np.array(embed_text(text))
        )

        return Ok(
            SuccessForReview(
                success_return=result.unwrap(), embedding=embedding, text=text
            )
        )
