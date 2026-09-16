"""Create Test Dataset
using dataset
https://huggingface.co/datasets/dmilush/shieldlm-prompt-injection

"""

import asyncio
import warnings
from pathlib import Path

import polars as pl
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
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
):

    from ml_factory.datasets import precompute_features_ae
    from ml_factory.utils import embedding_text

    file_name = file_name or raw_file_name
    output_path = output_path or processed_file_name

    file_name = build_parquet_file(file_name, exists_run=False)
    lazy_df = pl.scan_parquet(file_name)
    asyncio.run(
        precompute_features_ae(
            lazy_df,
            embedding_text,
            embedding_model_name=embedding_model,
            out_path=output_path,
            extra_columns=["label_category"],
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

        # optional: per-category breakdown, appended as additional rows
        if categories is not None:
            for category in sorted(set(categories) - {"benign"}):
                cat_mask = (categories == category) | (categories == "benign")
                cat_labels = labels[cat_mask]
                cat_scores = scores[cat_mask]
                cat_preds = predictions[cat_mask]

                if len(set(cat_labels)) < 2:
                    continue  # skip degenerate single-class slices

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


if __name__ == "__main__":
    trying_bert_dataset()
