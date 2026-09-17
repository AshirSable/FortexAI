"""
Stage 1 of the cascade: semantic search.

Wraps Backend/semantic_search/search.py (search_prompt / add_confirmed) as a
PipelinePhase. We don't change anything inside Backend/semantic_search/ - we
just call it from here.
"""

import numpy as np

from app.pipeline import PipelinePhase
from app.pipeline.utils import add_semantic_search_to_path, embed_text
from app.type_store import Err, Ok, Phase, PhaseInput, Result, SuccessForReview, SuccessReturn, Verdict
from app.type_store._error import InferenceError, PhaseError

add_semantic_search_to_path()
from search import add_confirmed, search_prompt  # semantic_search/search.py


class SemanticSearchPipeline(PipelinePhase):
    def __init__(self):
        super().__init__(phase=Phase.semantic_search)

    def verdict(self, input: PhaseInput) -> Result[SuccessReturn, PhaseError]:
        text = input.require_text()

        try:
            search_result = search_prompt(text)
        except Exception as e:
            return Err(InferenceError(f"semantic search failed: {e}"))

        match = search_result["match"]
        similarity = float(search_result["similarity"])

        if match == "attack":
            verdict = Verdict.attack
        elif match == "benign":
            verdict = Verdict.benign
        else:
            # "not_found" - nothing close enough in either index, so this
            # stage has no opinion. Let the next stage (autoencoder) decide.
            verdict = Verdict.undetermined

        return Ok(SuccessReturn(verdict=verdict, at_phase=self.phase, confidence=similarity))

    def verdict_with_data(self, input: PhaseInput) -> Result[SuccessForReview, PhaseError]:
        result = self.verdict(input)
        if result.is_err():
            return result

        text = input.require_text()
        embedding = input.embedding if input.embedding is not None else np.array(embed_text(text))

        return Ok(SuccessForReview(success_return=result.unwrap(), embedding=embedding, text=text))

    def confirm(self, text: str, label: str):
        """
        Records a confirmed verdict back into the semantic search index, so
        future identical/near-identical prompts get caught here next time.
        Thin pass-through to semantic_search/search.py's add_confirmed().
        """
        return add_confirmed(text, label)
