"""
Stage 3 of the cascade: Mini-BERT ensemble classifier.

Loads an ensemble of 5 trained Mini-BERT classifiers (bert_1.pt to bert_5.pt) from
MODEL_DIRECTORY_RELEASED / "ensemble_classifier_mini", tokenizes the prompt,
runs forward inference across all 5 models, averages the attack probabilities,
and determines the verdict based on the confidence threshold.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
from ml_factory import MODEL_DIRECTORY_RELEASED

MODEL_DIR = MODEL_DIRECTORY_RELEASED / "ensemble_classifier_mini"
FALLBACK_TOKENIZER_NAME = "google/bert_uncased_L-4_H-256_A-4"
FALLBACK_MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"

ATTACK_LABEL_INDEX = 1
NUM_MODELS = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EnsembleBERTPipeline(PipelinePhase):
    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        super().__init__(phase=Phase.ensemble_bert)

        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise ModelUnavailableError(
                f"Ensemble classifier directory not found at {self.model_dir}"
            )

        self.confidence_threshold = confidence_threshold

        # 1. Initialize Tokenizer (checking for local files vs fallback)
        has_real_tokenizer_files = (
            (self.model_dir / "tokenizer.json").exists()
            or (self.model_dir / "vocab.txt").exists()
        )
        tokenizer_source = self.model_dir if has_real_tokenizer_files else FALLBACK_TOKENIZER_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        # 2. Load the 5 individual BERT models
        self.models: List[torch.nn.Module] = []
        for i in range(1, NUM_MODELS + 1):
            model_path = self.model_dir / f"bert_{i}.pt"
            if not model_path.is_file():
                raise ModelUnavailableError(
                    f"Expected model checkpoint missing: {model_path}"
                )

            # Instantiate architecture using config / base weights and load checkpoint weights
            model = AutoModelForSequenceClassification.from_pretrained(
                FALLBACK_MODEL_NAME,
                num_labels=2,
            )
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            self.models.append(model)

    def attack_probability(self, text: str) -> float:
        """Tokenizes text, evaluates each ensemble member, and returns mean attack probability."""
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(DEVICE)

        model_attack_probs: List[float] = []

        with torch.no_grad():
            for model in self.models:
                outputs = model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
                probabilities = F.softmax(outputs.logits, dim=-1)[0]
                model_attack_probs.append(float(probabilities[ATTACK_LABEL_INDEX]))

        # Mean ensemble aggregation across the 5 models
        return float(np.mean(model_attack_probs))

    def verdict(self, input: PhaseInput) -> Result[SuccessReturn, PhaseError]:
        text = input.require_text()

        try:
            attack_prob = self.attack_probability(text)
        except Exception as e:
            return Err(InferenceError(f"Ensemble BERT inference failed: {e}"))

        benign_prob = 1.0 - attack_prob

        # Threshold checking:
        # A verdict requires the probability mass to be clear relative to the 0.5 center.
        # If the gap |P - 0.5| does not meet the confidence margin (0.25), classify as undetermined.
        if attack_prob >= (0.5 + self.confidence_threshold):
            verdict = Verdict.attack
            confidence = attack_prob
        elif attack_prob <= (0.5 - self.confidence_threshold):
            verdict = Verdict.benign
            confidence = benign_prob
        else:
            verdict = Verdict.undetermined
            confidence = max(attack_prob, benign_prob)

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