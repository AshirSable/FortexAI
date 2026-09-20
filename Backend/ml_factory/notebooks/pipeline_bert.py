"""
BERT Ensemble Inference Pipeline

Pipeline:
    Raw prompt
        |
        v
    BERT tokenizer
        |
        v
    Token chunks
    max_length=128
    stride=32
        |
        v
    5 trained BERT classifiers
        |
        v
    Softmax probability of class 1
        |
        v
    Average probabilities across 5 models
        |
        v
    Average probabilities across chunks
        |
        v
    Threshold = 0.25
        |
        v
    Binary prediction

Example:

    python pipeline.py \
        --prompt "Your prompt goes here"

Optional:

    python pipeline.py \
        --prompt "Your prompt goes here" \
        --threshold 0.25 \
        --max-length 128 \
        --stride 32 \
        --batch-size 32

Expected model files:

    ensemble_classifier_mini/
        bert_1.pt
        bert_2.pt
        bert_3.pt
        bert_4.pt
        bert_5.pt

The .pt files are expected to contain state_dicts compatible with:

    AutoModelForSequenceClassification.from_pretrained(
        "google/bert_uncased_L-4_H-256_A-4",
        num_labels=2
    )
"""

import sys


sys.path.append('../../')

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from ml_factory.datasets.precompute_tokens import precompute_tokens
from ml_factory.utils import merge_parts_to_dir
from ml_factory import DATA_RAW_DIR, DATA_PROCESSED_DIR
from pathlib import Path
from ml_factory.datasets import PromptBERTDataset
from ml_factory.datasets.sampler import SplitSampler
import numpy as np

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"

# Directory containing:
# bert_1.pt
# bert_2.pt
# ...
# bert_5.pt

MODEL_DIR = DATA_PROCESSED_DIR / "ensemble_classifier_mini"

NUM_MODELS = 5

MAX_LENGTH = 128
STRIDE = 32

BATCH_SIZE = 32

THRESHOLD = 0.25

# How to combine predictions when a prompt creates
# multiple overlapping chunks.
#
# "mean" is the recommended/default behavior here because
# it gives one probability representing the complete prompt.
CHUNK_AGGREGATION = "mean"


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Argument parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BERT ensemble inference on a single prompt."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt to classify.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory containing bert_1.pt ... bert_5.pt",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Classification threshold for class 1.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Maximum token sequence length.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help="Stride used for overlapping chunks.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Inference batch size.",
    )

    return parser.parse_args()


# ============================================================
# Load tokenizer
# ============================================================

def load_tokenizer():
    print("\nLoading tokenizer...")
    print(f"Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    return tokenizer


# ============================================================
# Tokenize and chunk prompt
# ============================================================

def tokenize_and_chunk(
    prompt,
    tokenizer,
    max_length=128,
    stride=32,
):
    """
    Tokenize a prompt and split it into overlapping chunks.

    This is intended to reproduce the important behavior of:

        precompute_tokens(
            token_chunk=True,
            stride=32,
            max_length=128
        )

    Each chunk contains:
        input_ids
        attention_mask

    Special tokens ([CLS], [SEP]) are added by the tokenizer.

    Returns
    -------
    input_ids : torch.Tensor
        Shape: [num_chunks, max_length]

    attention_masks : torch.Tensor
        Shape: [num_chunks, max_length]
    """

    if not prompt or not prompt.strip():
        raise ValueError(
            "The prompt is empty. Please provide a non-empty prompt."
        )

    if stride >= max_length:
        raise ValueError(
            f"stride ({stride}) must be smaller than "
            f"max_length ({max_length})."
        )

    # --------------------------------------------------------
    # Tokenize with overflowing chunks
    # --------------------------------------------------------
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_masks = encoded["attention_mask"]

    return input_ids, attention_masks


# ============================================================
# Load ensemble models
# ============================================================

def load_models(model_dir, device):
    """
    Load all five trained models.

    Each model is initialized from the same BERT architecture
    and then its trained state_dict is loaded from:

        bert_1.pt
        ...
        bert_5.pt
    """

    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}"
        )

    models = []

    print("\nLoading ensemble models...")
    print(f"Model directory: {model_dir}")
    print(f"Device: {device}")

    for i in range(1, NUM_MODELS + 1):

        model_path = model_dir / f"bert_{i}.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Could not find model file: {model_path}"
            )

        print(f"Loading model {i}: {model_path}")

        # Same architecture used in your OOD testing code.
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )

        state_dict = torch.load(
            model_path,
            map_location=device,
        )

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        models.append(model)

    print(f"Successfully loaded {len(models)} models.")

    return models


# ============================================================
# Create DataLoader
# ============================================================

def create_dataloader(
    input_ids,
    attention_masks,
    batch_size,
):
    """
    Create a DataLoader for tokenized chunks.

    This mirrors the batching behavior of your
    PromptBERTDataset/DataLoader inference pipeline,
    but does not require writing the prompt to parquet.
    """

    dataset = TensorDataset(
        input_ids,
        attention_masks,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return loader


# ============================================================
# Run one model
# ============================================================

def predict_single_model(
    model,
    dataloader,
    device,
):
    """
    Run one BERT model over all chunks.

    Returns:
        numpy array containing probability of class 1
        for every chunk.
    """

    probabilities = []

    with torch.inference_mode():

        for input_ids, attention_masks in dataloader:

            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
            )

            # ------------------------------------------------
            # Probability of class 1
            #
            # This exactly follows your OOD testing code:
            #
            # torch.softmax(outputs.logits, dim=1)[:, 1]
            # ------------------------------------------------
            batch_probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )[:, 1]

            probabilities.extend(
                batch_probabilities.cpu().numpy()
            )

    return np.asarray(probabilities)


# ============================================================
# Aggregate ensemble probabilities
# ============================================================

def aggregate_models(all_model_probabilities):
    """
    Average probabilities from all five models.

    Input:
        Shape:
            [5, num_chunks]

    Output:
        Shape:
            [num_chunks]
    """

    probability_matrix = np.vstack(
        all_model_probabilities
    )

    ensemble_probabilities = np.mean(
        probability_matrix,
        axis=0,
    )

    return ensemble_probabilities, probability_matrix


# ============================================================
# Aggregate chunks
# ============================================================

def aggregate_chunks(
    chunk_probabilities,
    method="mean",
):
    """
    Convert chunk-level probabilities into one
    prompt-level probability.

    Supported methods:
        mean
        max
    """

    if len(chunk_probabilities) == 0:
        raise ValueError(
            "No chunk probabilities were produced."
        )

    if method == "mean":

        prompt_probability = float(
            np.mean(chunk_probabilities)
        )

    elif method == "max":

        prompt_probability = float(
            np.max(chunk_probabilities)
        )

    else:

        raise ValueError(
            f"Unknown chunk aggregation method: {method}"
        )

    return prompt_probability


# ============================================================
# Main inference pipeline
# ============================================================

def run_pipeline(
    prompt,
    model_dir,
    threshold,
    max_length,
    stride,
    batch_size,
):
    """
    Complete end-to-end inference pipeline.

    Returns a dictionary containing:
        prediction
        probability
        number of chunks
        model probabilities
        chunk probabilities
        latency
    """

    # --------------------------------------------------------
    # Start total latency timer
    #
    # This includes:
    #   tokenizer loading
    #   model loading
    #   tokenization
    #   chunking
    #   DataLoader creation
    #   inference
    #   aggregation
    #
    # Therefore this represents TOTAL pipeline latency.
    # --------------------------------------------------------

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer_start = time.perf_counter()

    tokenizer = load_tokenizer()

    tokenizer_load_time = (
        time.perf_counter() - tokenizer_start
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    model_start = time.perf_counter()

    models = load_models(
        model_dir=model_dir,
        device=DEVICE,
    )

    model_load_time = (
        time.perf_counter() - model_start
    )

    # --------------------------------------------------------
    # Tokenization + chunking
    # --------------------------------------------------------

    preprocessing_start = time.perf_counter()

    input_ids, attention_masks = tokenize_and_chunk(
        prompt=prompt,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )

    preprocessing_time = (
        time.perf_counter() - preprocessing_start
    )

    number_of_chunks = input_ids.shape[0]

    print("\nPrompt preprocessing")
    print("--------------------")
    print(f"Number of chunks: {number_of_chunks}")
    print(f"Max length      : {max_length}")
    print(f"Stride          : {stride}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataloader = create_dataloader(
        input_ids=input_ids,
        attention_masks=attention_masks,
        batch_size=batch_size,
    )

    # --------------------------------------------------------
    # Run all five models
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    all_model_probabilities = []

    for model_idx, model in enumerate(models):

        model_probabilities = predict_single_model(
            model=model,
            dataloader=dataloader,
            device=DEVICE,
        )

        all_model_probabilities.append(
            model_probabilities
        )

        print(
            f"Model {model_idx + 1}: "
            f"{len(model_probabilities)} chunk predictions"
        )

    inference_time = (
        time.perf_counter() - inference_start
    )

    # --------------------------------------------------------
    # Ensemble aggregation
    # --------------------------------------------------------

    aggregation_start = time.perf_counter()

    ensemble_probabilities, probability_matrix = (
        aggregate_models(
            all_model_probabilities
        )
    )

    # --------------------------------------------------------
    # Aggregate chunks to obtain one prompt probability
    # --------------------------------------------------------

    prompt_probability = aggregate_chunks(
        ensemble_probabilities,
        method=CHUNK_AGGREGATION,
    )

    # --------------------------------------------------------
    # Final binary classification
    # --------------------------------------------------------

    prediction = int(
        prompt_probability >= threshold
    )

    aggregation_time = (
        time.perf_counter() - aggregation_start
    )

    # --------------------------------------------------------
    # Total latency
    # --------------------------------------------------------

    total_latency = (
        time.perf_counter() - total_start
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BERT ENSEMBLE INFERENCE RESULT")
    print("=" * 60)

    print(f"\nPrediction      : {prediction}")
    print(f"Probability     : {prompt_probability:.6f}")
    print(f"Threshold       : {threshold:.6f}")

    if prediction == 1:
        print("Class           : 1")
    else:
        print("Class           : 0")

    print("\n" + "-" * 60)
    print("Chunk-level ensemble probabilities")
    print("-" * 60)

    for chunk_idx, probability in enumerate(
        ensemble_probabilities,
        start=1,
    ):

        print(
            f"Chunk {chunk_idx}: "
            f"{probability:.6f}"
        )

    print("\n" + "-" * 60)
    print("Model probabilities")
    print("-" * 60)

    for model_idx in range(NUM_MODELS):

        model_probs = probability_matrix[model_idx]

        print(
            f"Model {model_idx + 1}: "
            f"mean={np.mean(model_probs):.6f}, "
            f"min={np.min(model_probs):.6f}, "
            f"max={np.max(model_probs):.6f}"
        )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("Latency")
    print("-" * 60)

    print(
        f"Tokenizer loading : "
        f"{tokenizer_load_time * 1000:.2f} ms"
    )

    print(
        f"Model loading     : "
        f"{model_load_time * 1000:.2f} ms"
    )

    print(
        f"Preprocessing     : "
        f"{preprocessing_time * 1000:.2f} ms"
    )

    print(
        f"Inference         : "
        f"{inference_time * 1000:.2f} ms"
    )

    print(
        f"Aggregation       : "
        f"{aggregation_time * 1000:.2f} ms"
    )

    print(
        f"TOTAL LATENCY     : "
        f"{total_latency * 1000:.2f} ms"
    )

    print(
        f"TOTAL LATENCY     : "
        f"{total_latency:.4f} seconds"
    )

    print("=" * 60)

    return {
        "prediction": prediction,
        "probability": prompt_probability,
        "threshold": threshold,
        "number_of_chunks": number_of_chunks,
        "chunk_probabilities": ensemble_probabilities,
        "model_probability_matrix": probability_matrix,
        "tokenizer_load_time": tokenizer_load_time,
        "model_load_time": model_load_time,
        "preprocessing_time": preprocessing_time,
        "inference_time": inference_time,
        "aggregation_time": aggregation_time,
        "total_latency": total_latency,
    }


# ============================================================
# Entry point
# ============================================================

def main():

    args = parse_args()

    print("=" * 60)
    print("BERT ENSEMBLE PIPELINE")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    # --------------------------------------------------------
    # CUDA information
    # --------------------------------------------------------

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version: {torch.version.cuda}"
        )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    run_pipeline(
        prompt=args.prompt,
        model_dir=args.model_dir,
        threshold=args.threshold,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
