"""
Stage 2 of the cascade: autoencoder anomaly check.

Loads the trained NormalityAE checkpoint and flags a prompt as benign/attack
based on how well the autoencoder can reconstruct it - a high reconstruction
error means the prompt looks "unusual" compared to what the model was
trained on.

IMPORTANT / GUESSED PIECES (see summary for full detail):
- The checkpoint at MODEL_PATH below has no saved threshold and no recorded
  training config next to it, so LOW_ERROR_THRESHOLD / HIGH_ERROR_THRESHOLD
  are placeholder guesses, not calibrated values. They need to be tuned
  against a real validation set before this stage is trusted.
- The model's input is not the raw 768-dim embedding. The checkpoint's first
  layer expects 777 inputs = 768-dim nomic-embed-text embedding + 9
  "structural" features (word count, sentence length, etc. - see
  ml_factory/utils/structural_extractor.py), concatenated embedding-first
  (that's the order ml_factory/datasets/__init__.py uses when it builds
  training data, and it's what AE_INPUT_DIM/AE_ARGS below assume).
"""

from pathlib import Path

import numpy as np
import torch
from ml_factory import MODEL_DIRECTORY_RELEASED
from ml_factory.models import Args
from ml_factory.models.autoencoder import NormalityAE
from ml_factory.utils.scaler import StandardScaler
from ml_factory.utils.structural_extractor import StructuralExtractor

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
from app.type_store._error import InferenceError, ModelUnavailableError, PhaseError

# the only autoencoder checkpoint currently in the repo (added 2026-09-17).
MODEL_PATH = MODEL_DIRECTORY_RELEASED / "NormalityAE_instance_1789824349.7443838"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# the StandardScaler (per-feature mean/std) fit on the training data before
# training this exact checkpoint - see ml_factory/training_scripts/scripts_normality_ae.py,
# which standardizes the 777-dim embedding+structural vector before it ever
# reaches the model. Without this, reconstruction error is dominated by the
# raw (unstandardized) structural features and is meaningless.
SCALER_PATH = MODEL_DIRECTORY_RELEASED / "SCALER_FILE_FOR_FULL_SEED_3123.pt"

# matches the checkpoint's tensor shapes exactly (see summary): a 768-dim
# embedding plus 9 structural features, 8 experts, top-2 bottleneck of 64.
# expert_k=2 is the Args default and can't be verified from the checkpoint's
# weight shapes alone (it only affects the forward pass, not layer sizes).
EMBEDDING_DIM = 768
AE_INPUT_DIM = EMBEDDING_DIM + 9
AE_ARGS = Args(ae_bottleneck=100, n_experts=8, expert_k=4)


#  One of calibrated values,

# Calibrated Values
LOW_ERROR_THRESHOLD = 1.0069705247879028  # reconstruction error at or below this -> benign #  # Mean Reconstruction loss for Benign data in OOD Validation Dataset
HIGH_ERROR_THRESHOLD = 2.154469  # reconstruction error at or above this -> attack #  # Threshold with the lowes TPR ~ 0.502 (Detecting Attack Correctly) and Highest TNR ~ 0.985 (Detecting Benign)
# anything in between -> undetermined, pass to the next stage

# TODO: Need to calibrate the low error and high error threshold for this model


class AutoEncoderPipeline(PipelinePhase):
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        low_error_threshold: float = LOW_ERROR_THRESHOLD,
        high_error_threshold: float = HIGH_ERROR_THRESHOLD,
    ):
        super().__init__(phase=Phase.autoencoder)

        if not Path(model_path).is_file():
            raise ModelUnavailableError(
                f"autoencoder checkpoint not found at {model_path}"
            )

        if not Path(scaler_path).is_file():
            raise ModelUnavailableError(
                f"autoencoder standard scaler not found at {scaler_path}"
            )

        self.low_error_threshold = low_error_threshold
        self.high_error_threshold = high_error_threshold
        self.structural_extractor = StructuralExtractor()
        self.standard_scaler = StandardScaler.load(scaler_path, device=DEVICE)

        self.model = NormalityAE(input_dim=AE_INPUT_DIM, args=AE_ARGS, _mode_2=True)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(DEVICE)
        self.model.eval()

    def build_input_vector(self, text: str, embedding) -> torch.Tensor:
        """Turns prompt text (+ optional pre-computed embedding) into the 777-dim vector the model expects."""
        if embedding is None:
            embedding = embed_text(text)

        embedding = np.array(embedding, dtype="float32")
        structural = np.array(
            self.structural_extractor.structural_fn(text), dtype="float32"
        )
        full_vector = np.concatenate([embedding, structural])

        full_vector = torch.from_numpy(full_vector).to(DEVICE)
        full_vector = self.standard_scaler.transform(full_vector)

        # model expects a batch, so add a batch dimension of size 1
        return full_vector.unsqueeze(0)

    def reconstruction_error(self, text: str, embedding) -> float:
        input_vector = self.build_input_vector(text, embedding)
        with torch.no_grad():
            result = self.model(input_vector)
            error = torch.mean((result.recon - input_vector) ** 2)
        return float(error.item())

    def verdict(self, input: PhaseInput) -> Result[SuccessReturn, PhaseError]:
        text = input.require_text()

        try:
            error = self.reconstruction_error(text, input.embedding)
        except Exception as e:
            return Err(InferenceError(f"autoencoder inference failed: {e}"))

        if error <= self.low_error_threshold:
            verdict = Verdict.benign
            # error near 0 -> confidence near 1; error near the threshold -> confidence near 0
            confidence = max(0.0, min(1.0, 1.0 - (error / self.low_error_threshold)))
        elif error >= self.high_error_threshold:
            verdict = Verdict.attack
            # error at the threshold -> confidence near 0; error far past it -> confidence near 1
            confidence = max(0.0, min(1.0, error / (self.high_error_threshold * 2)))
        else:
            verdict = Verdict.undetermined
            confidence = 0.0

        return Ok(
            SuccessReturn(verdict=verdict, at_phase=self.phase, confidence=confidence)
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
