"""
Stage 4 of the cascade: LLM judge. This is the LAST stage - if a prompt gets
here, every earlier stage (semantic search, autoencoder, mini-BERT) looked at
it and had no opinion.

Which LLM actually does the judging is still an open decision (separate,
ongoing work comparing a local llama3.1:8b vs. Groq's openai/gpt-oss-20b), so
there is no real model call here yet.

Until that's wired in, this stage fails CLOSED: it reports "attack" with low
confidence, rather than defaulting to "benign". A prompt that got this far
without being cleared should not be let through just because we haven't
picked a judge model yet.
"""

import numpy as np

from app.pipeline import PipelinePhase
from app.pipeline.utils import embed_text
from app.type_store import Ok, Phase, PhaseInput, Result, SuccessForReview, SuccessReturn, Verdict
from app.type_store._error import PhaseError

# low on purpose: this isn't a real judgement, just a safe default
FAIL_CLOSED_CONFIDENCE = 0.1


class LLM_JudgePipeline(PipelinePhase):
    def __init__(self):
        super().__init__(phase=Phase.llm_judge)

    def verdict(self, input: PhaseInput) -> Result[SuccessReturn, PhaseError]:
        # TODO: replace this once the judge model is chosen. It should call
        # the real LLM judge and return Verdict.benign / Verdict.attack based
        # on what it says. Until then: fail closed.
        return Ok(SuccessReturn(verdict=Verdict.attack, at_phase=self.phase, confidence=FAIL_CLOSED_CONFIDENCE))

    def verdict_with_data(self, input: PhaseInput) -> Result[SuccessForReview, PhaseError]:
        result = self.verdict(input)
        if result.is_err():
            return result

        text = input.require_text()
        embedding = input.embedding if input.embedding is not None else np.array(embed_text(text))

        return Ok(SuccessForReview(success_return=result.unwrap(), embedding=embedding, text=text))
