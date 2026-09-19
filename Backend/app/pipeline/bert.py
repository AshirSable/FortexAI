"""
Stage 3 of the cascade: Mini-BERT classifier.

Loads the trained Mini-BERT classifier from
Backend/ml_factory/notebooks/best_bert_classifier/ (HuggingFace format:
config.json + model.safetensors) and classifies the prompt text directly.

GUESSED PIECE: that folder only has config.json + model.safetensors, no
tokenizer files (no vocab.txt / tokenizer.json / tokenizer_config.json). The
training notebook (ml_factory/notebooks/bert_single.ipynb) fine-tunes from
"google/bert_uncased_L-2_H-128_A-2", and that base model's config
(hidden_size=128, num_hidden_layers=2, num_attention_heads=2) matches
best_bert_classifier/config.json exactly, so that's used as the tokenizer.

IMPORTANT: don't try `AutoTokenizer.from_pretrained(MODEL_DIR)` and fall back
on an exception. It does NOT raise when the folder has no vocab file - it
silently builds a near-empty tokenizer (vocab_size=5) that maps almost every
word to [UNK], which quietly makes the classifier useless (verified while
testing this: it scored an obvious attack prompt and an obviously harmless
one almost identically, ~0.96, because both turned into a wall of [UNK]
tokens). So we check for real tokenizer files ourselves instead of trusting
from_pretrained to fail loudly when they're missing.
"""

from pathlib import Path

import numpy as np
import torch
from ml_factory import MODEL_DIRECTORY_RELEASED
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

MODEL_DIR = MODEL_DIRECTORY_RELEASED
FALLBACK_TOKENIZER_NAME = "google/bert_uncased_L-2_H-128_A-2"

# label 1 = attack, label 0 = benign, matching how the model was trained
# (see ml_factory/datasets/test.py's f1_score(..., pos_label=1) for the attack class)
ATTACK_LABEL_INDEX = 1

# below this confidence in either direction, the stage has no opinion
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class EnsembleBERTPipeline(PipelinePhase):
    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        super().__init__(phase=Phase.ensemble_bert)

        if not Path(model_dir).is_dir():
            raise ModelUnavailableError(
                f"bert classifier folder not found at {model_dir}"
            )

        self.confidence_threshold = confidence_threshold
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        model_dir = Path(model_dir)
        has_real_tokenizer_files = (model_dir / "tokenizer.json").exists() or (
            model_dir / "vocab.txt"
        ).exists()

        if has_real_tokenizer_files:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER_NAME)

    def attack_probability(self, text: str) -> float:
        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self.model(**tokens).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        return float(probabilities[ATTACK_LABEL_INDEX])

    def verdict(self, input: PhaseInput) -> Result[SuccessReturn, PhaseError]:
        text = input.require_text()

        try:
            attack_probability = self.attack_probability(text)
        except Exception as e:
            return Err(InferenceError(f"bert classifier inference failed: {e}"))

        benign_probability = 1.0 - attack_probability

        if attack_probability >= self.confidence_threshold:
            verdict = Verdict.attack
            confidence = attack_probability
        elif benign_probability >= self.confidence_threshold:
            verdict = Verdict.benign
            confidence = benign_probability
        else:
            verdict = Verdict.undetermined
            confidence = max(attack_probability, benign_probability)

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
