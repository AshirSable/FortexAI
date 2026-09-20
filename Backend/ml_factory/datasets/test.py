"""Create Test Dataset
using dataset
https://huggingface.co/datasets/dmilush/shieldlm-prompt-injection

"""

import asyncio
import warnings
from pathlib import Path

import polars as pl
import torch
from datasets.utils.py_utils import Literal
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from ml_factory import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODEL_DIRECTORY_DEV
from ml_factory.datasets import PromptBERTDataset, PromptFeatureDataset
from ml_factory.datasets.precompute_tokens import precompute_tokens
from ml_factory.datasets.sampler import SplitSampler
from ml_factory.models.autoencoder import ModelResult
from ml_factory.utils import merge_parts_to_dir
from ml_factory.utils.scaler import StandardScaler
from ml_factory.utils.structural_extractor import StructuralExtractor

raw_file_name = DATA_RAW_DIR / "shieldlm_data_test.parquet"

processed_file_name = DATA_PROCESSED_DIR / "shieldlm_data_test_processed.npz"


def trying_bert_dataset():
    full_data_path = DATA_RAW_DIR / "final_data.parquet"
    full_process_path_dir = (
        DATA_PROCESSED_DIR / "final_data_autotokenizer_googlebert_chunking_32_stride/"
    )
    paths = precompute_tokens(
        dataset_path=full_data_path,
        out_path=full_process_path_dir,
        tokenizer=AutoTokenizer,
        pretrained_tokenizer="google/bert_uncased_L-2_H-128_A-2",
        max_length=128,
        write_id_to_dataset=True,
        save_bytes=100_000_000,
        chunk_size=256,
        token_chunk=True,
        stride=32,
    )
    sampler = SplitSampler(train_split=0.7, test_split=0.15, validation_split=0.15)

    merge_parts_to_dir(paths, full_process_path_dir)

    dataset = PromptBERTDataset(full_process_path_dir)

    sampler = sampler.build_split(dataset.item_ids, labels=dataset.labels)

    train_dataset = Subset(dataset, dataset.ids_to_position(sampler.get_split("train")))

    train_loader = DataLoader(train_dataset)

    for data in train_loader:
        print(data)
        break


def build_parquet_file(file: Path, exists_run: bool = False) -> Path:
    if exists_run or not file.is_file():
        splits = {
            "train": "data/train-00000-of-00001.parquet",
            "validation": "data/validation-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        }
        df_train = pl.scan_parquet(
            "hf://datasets/dmilush/shieldlm-prompt-injection/" + splits["train"]
        )
        df_validation = pl.scan_parquet(
            "hf://datasets/dmilush/shieldlm-prompt-injection/" + splits["validation"]
        )
        df_test = pl.scan_parquet(
            "hf://datasets/dmilush/shieldlm-prompt-injection/" + splits["test"]
        )

        pl.concat([df_train, df_validation, df_test]).rename(
            {"label_binary": "label"}
        ).sink_parquet(file)

    return file


def process_data(
    file_name: Path | None = None,
    output_path: Path | None = None,
    embedding_model: str = "nomic-embed-text",
    structural_fn: StructuralExtractor | None = None,
    save_n: int | Literal["all"] = "all",
) -> list:

    from ml_factory.datasets.precompute_features import precompute_features_ae
    from ml_factory.utils import embedding_text

    file_name = file_name or raw_file_name
    output_path = output_path or processed_file_name

    file_name = build_parquet_file(file_name, exists_run=False)
    lazy_df = pl.scan_parquet(file_name)
    return asyncio.run(
        precompute_features_ae(
            lazy_df,
            embedding_text,
            embedding_model_name=embedding_model,
            out_path=output_path,
            extra_columns=["label_category"],
            structural_extractor=structural_fn,
            batch_size=256,
            chunk_size=50,
            save_n=save_n,
        )
    )


def evaluate_on_shieldlm_test_single(
    model: torch.nn.Module,
    threshold: float,
    loader: DataLoader,
    device: str = "cuda",
):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            result = model(X)
            if isinstance(result, ModelResult):
                scores = ((result.recon - X) ** 2).mean(dim=1)
            else:
                scores = ((result[0] - X) ** 2).mean(dim=1)
            all_scores.append(scores.cpu())
            all_labels.append(y.cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    predictions = (scores >= threshold).astype(int)

    return scores, labels, predictions


def evaluate_ood_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    model_name: str,
    threshold: float | None = None,
    device: str = "cuda",
    categories: np.ndarray | None = None,
) -> pl.DataFrame:
    """
    Wraps evaluate_on_shieldlm_test_single to compute standard OOD detection
    metrics and return them as a polars DataFrame.

    Threshold-independent metrics (AUPRC, ROC-AUC) are always computed.
    Threshold-dependent metrics (accuracy, F1, precision, recall, confusion
    matrix counts) are only computed if `threshold` is provided.

    If `categories` is provided (array aligned with the loader's samples,
    e.g. dataset.extra["label_category"]), an additional row is returned
    per attack category, computed against that category's samples plus all
    benign samples — mirroring evaluate_on_shieldlm_test's breakdown.
    """
    model = model.to(device)
    scores, labels, predictions = evaluate_on_shieldlm_test_single(
        model=model,
        threshold=threshold if threshold is not None else 0.0,
        loader=loader,
        device=device,
    )

    def _compute_row(row_name: str, scores_, labels_, predictions_) -> dict:
        n_benign = int((labels_ == 0).sum())
        n_attack = int((labels_ == 1).sum())

        row: dict = {
            "model_name": row_name,
            "n_samples": len(labels_),
            "n_benign": n_benign,
            "n_attack": n_attack,
            "threshold": threshold,
            "roc_auc": roc_auc_score(labels_, scores_),
            "auprc": average_precision_score(labels_, scores_),
            "score_mean_benign": (
                float(scores_[labels_ == 0].mean()) if n_benign > 0 else None
            ),
            "score_mean_attack": (
                float(scores_[labels_ == 1].mean()) if n_attack > 0 else None
            ),
            "score_std_benign": (
                float(scores_[labels_ == 0].std()) if n_benign > 0 else None
            ),
            "score_std_attack": (
                float(scores_[labels_ == 1].std()) if n_attack > 0 else None
            ),
        }

        if threshold is not None:
            tn, fp, fn, tp = confusion_matrix(
                labels_, predictions_, labels=[0, 1]
            ).ravel()
            row.update(
                {
                    "accuracy": accuracy_score(labels_, predictions_),
                    "precision": precision_score(
                        labels_, predictions_, zero_division=0
                    ),
                    "recall": recall_score(labels_, predictions_, zero_division=0),
                    "f1": f1_score(labels_, predictions_, zero_division=0),
                    "fpr": fp / (fp + tn) if (fp + tn) > 0 else None,
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn),
                }
            )
        return row

    rows = [_compute_row(model_name, scores, labels, predictions)]

    if categories is not None:
        categories = np.asarray(categories)
        for category in sorted(set(categories) - {"benign"}):
            cat_mask = (categories == category) | (categories == "benign")
            cat_labels = labels[cat_mask]

            if len(set(cat_labels)) < 2:
                continue  # need both classes present to compute AUPRC/ROC-AUC

            rows.append(
                _compute_row(
                    f"{model_name} [{category}]",
                    scores[cat_mask],
                    cat_labels,
                    predictions[cat_mask],
                )
            )

    return pl.DataFrame(rows)


def evaluate_on_shieldlm_test(
    model: nn.Module,
    model_files: dict[str, str],
    thresholds: dict[str, float],
    device: str = "cuda",
    batch_size: int = 128,
    processed_file: Path | None = None,
    raw_file: Path | None = None,
    scaler_file: Path | None = None,
    embedding_model: str = "nomic-embed-text",
    category_breakdown: bool = True,
) -> pl.DataFrame:

    processed_file = processed_file or processed_file_name
    raw_file = raw_file or raw_file_name
    if not processed_file.is_file():
        process_data(raw_file, processed_file, embedding_model)

    dataset = PromptFeatureDataset(processed_file)

    if scaler_file is None:
        warnings.warn(
            "no scaler file provided, results may not be optimal", category=UserWarning
        )
    else:
        scaler = StandardScaler.load(scaler_file)
        dataset.features = scaler.transform(dataset.features)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    categories = dataset.extra.get("label_category") if category_breakdown else None

    rows = []
    for variant, file_name in model_files.items():
        checkpoint = torch.load(MODEL_DIRECTORY_DEV / file_name, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.to(device)

        threshold = thresholds[variant]
        scores, labels, predictions = evaluate_on_shieldlm_test_single(
            model, threshold, loader, device=device
        )

        rows.append(
            {
                "Variant": variant,
                "Threshold": threshold,
                "ROC-AUC": roc_auc_score(labels, scores),
                "Avg. Precision": average_precision_score(labels, scores),
                "Attack F1": f1_score(labels, predictions, pos_label=1),
                "Balanced Acc.": balanced_accuracy_score(labels, predictions),
            }
        )

        if categories is not None:
            for category in sorted(set(categories) - {"benign"}):
                cat_mask = (categories == category) | (categories == "benign")
                cat_labels = labels[cat_mask]
                cat_scores = scores[cat_mask]
                cat_preds = predictions[cat_mask]

                if len(set(cat_labels)) < 2:
                    continue

                rows.append(
                    {
                        "Variant": f"{variant} [{category}]",
                        "Threshold": threshold,
                        "ROC-AUC": roc_auc_score(cat_labels, cat_scores),
                        "Avg. Precision": average_precision_score(
                            cat_labels, cat_scores
                        ),
                        "Attack F1": f1_score(cat_labels, cat_preds, pos_label=1),
                        "Balanced Acc.": balanced_accuracy_score(cat_labels, cat_preds),
                    }
                )

    return pl.DataFrame(rows)


def compute_threshold_from_benign(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    percentile: float = 95.0,
) -> float:
    """
    Computes an anomaly-score threshold from a benign-only (or mixed, but
    only benign samples are used) validation set, as the given percentile
    of the benign reconstruction-error distribution.

    Per the earlier discussion: threshold selection should be derived from
    benign data only, never fit directly against OOD/attack samples, to
    avoid overfitting the threshold to specific attack types seen at
    selection time.
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            result = model(X)
            if isinstance(result, ModelResult):
                scores = ((result.recon - X) ** 2).mean(dim=1)
            else:
                scores = ((result[0] - X) ** 2).mean(dim=1)
            all_scores.append(scores.cpu())
            all_labels.append(y.cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    benign_scores = scores[labels == 0]
    if len(benign_scores) == 0:
        raise ValueError(
            "No benign (label == 0) samples found in val_loader — "
            "threshold must be derived from benign data."
        )

    threshold = float(np.percentile(benign_scores, percentile))
    print(
        f"Threshold @ {percentile}th percentile of benign val scores "
        f"(n={len(benign_scores)}): {threshold:.6f}"
    )
    return threshold


def sweep_thresholds_from_benign(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
    percentiles: list[float] = [90.0, 95.0, 97.5, 99.0],
) -> dict[float, float]:
    """
    Same as compute_threshold_from_benign, but computes several candidate
    thresholds in one pass over val_loader (avoids re-running the model
    once per percentile). Returns {percentile: threshold}.
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            result = model(X)
            if isinstance(result, ModelResult):
                scores = ((result.recon - X) ** 2).mean(dim=1)
            else:
                scores = ((result[0] - X) ** 2).mean(dim=1)
            all_scores.append(scores.cpu())
            all_labels.append(y.cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    benign_scores = scores[labels == 0]

    if len(benign_scores) == 0:
        raise ValueError("No benign samples found in val_loader.")

    thresholds = {p: float(np.percentile(benign_scores, p)) for p in percentiles}
    for p, t in thresholds.items():
        print(f"  {p}th percentile: {t:.6f}")

    return thresholds


from sklearn.metrics import roc_curve


def compute_threshold_from_roc(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str = "cuda",
) -> tuple[float, dict]:
    """
    Computes a threshold via Youden's J statistic (argmax of TPR - FPR)
    on the ROC curve, using a labeled (benign + attack) validation set.
    """
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            result = model(X)
            if isinstance(result, ModelResult):
                scores = ((result.recon - X) ** 2).mean(dim=1)
            else:
                scores = ((result[0] - X) ** 2).mean(dim=1)
            all_scores.append(scores.cpu())
            all_labels.append(y.cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()

    threshold = float(thresholds[best_idx])
    diagnostics = {
        "threshold": threshold,
        "tpr": float(tpr[best_idx]),
        "fpr": float(fpr[best_idx]),
        "youden_j": float(j_scores[best_idx]),
    }

    print(
        f"Youden's J threshold: {threshold:.6f} "
        f"(TPR={diagnostics['tpr']:.4f}, FPR={diagnostics['fpr']:.4f}, "
        f"J={diagnostics['youden_j']:.4f})"
    )

    return threshold, diagnostics


if __name__ == "__main__":
    trying_bert_dataset()
