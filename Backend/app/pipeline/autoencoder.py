from ml_factory.models.autoencoder import BaseNormalAutoEncoder, NormalityAE

from app.pipeline import PipelinePhase
from app.type_store import Phase, PhaseInput, Result, SuccessForReview, SuccessReturn
from app.type_store._error import PhaseError


class AutoEncoderPipeline(PipelinePhase):
    def __init__(self, model_weights):
        super().__init__(phase=Phase("autoencoder"))
        ...

    def verdict(self, input: PhaseInput) -> Result[SuccessReturn, PhaseError]:
        return super().verdict(input)

    def verdict_with_data(
        self, input: PhaseInput
    ) -> Result[SuccessForReview, PhaseError]:
        return super().verdict_with_data(input)
