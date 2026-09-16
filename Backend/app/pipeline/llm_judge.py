from app.pipeline import PipelinePhase
from app.type_store import Phase, PhaseInput, Result, SuccessForReview, SuccessReturn
from app.type_store._error import PhaseError


class LLM_JudgePipeline(PipelinePhase):
    def __init__(self):
        super().__init__(phase=Phase("llm_judge"))

    def verdict(self, input: PhaseInput) -> Result[SuccessReturn, PhaseError]:
        return super().verdict(input)

    def verdict_with_data(
        self, input: PhaseInput
    ) -> Result[SuccessForReview, PhaseError]:
        return super().verdict_with_data(input)
