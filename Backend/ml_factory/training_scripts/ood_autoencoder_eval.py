from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, Subset

import ml_factory.datasets.test as ood_data
from ml_factory import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MODEL_DIRECTORY_DEV,
    TRAINING_LOGS_DIR,
)
from ml_factory.datasets import PromptFeatureDataset
from ml_factory.datasets.sampler import SplitSampler
from ml_factory.models import Args
from ml_factory.models.autoencoder import (
    BaseNormalAutoEncoder,
    NormalityAE,
    ModelResult,
)
from ml_factory.utils import merge_parts, merge_parts_simple
from ml_factory.utils.scaler import StandardScaler
from ml_factory.utils.structural_extractor import StructuralExtractor

import numpy as np


def evaluate_model_with_all_thresholds(
    model: torch.nn.Module,
    val_loader: DataLoader,
    dataloader: DataLoader,
    categories,
    out_path: Path,
) -> pl.DataFrame:
    model_name = model.__class__.__name__
    all_results = []

    # percentile-of-benign thresholds
    percentile_thresholds = ood_data.sweep_thresholds_from_benign(
        model=model,
        val_loader=val_loader,
        device="cuda",
        percentiles=[90.0, 95.0, 97.5, 99.0],
    )
    for pct, thresh in percentile_thresholds.items():
        res = ood_data.evaluate_ood_metrics(
            model=model,
            loader=dataloader,
            model_name=f"{model_name} @ p{pct}",
            threshold=thresh,
            device="cuda",
            categories=categories,
        )
        all_results.append(res)

    # Youden's J (ROC-based) threshold
    roc_threshold, roc_diag = ood_data.compute_threshold_from_roc(
        model=model,
        val_loader=val_loader,
        device="cuda",
    )
    res_roc = ood_data.evaluate_ood_metrics(
        model=model,
        loader=dataloader,
        model_name=f"{model_name} @ youden_j",
        threshold=roc_threshold,
        device="cuda",
        categories=categories,
    )
    all_results.append(res_roc)

    full_table = pl.concat(all_results)
    full_table.write_csv(out_path)
    return full_table


def sweep_low_threshold_by_std(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    std_multipliers: np.ndarray | None = None,
) -> pl.DataFrame:
    """
    Runs the model once over `loader`, then sweeps candidate low thresholds
    as score_mean_benign + k * score_std_benign for k in std_multipliers,
    reporting how many attacks would be bypassed (scored below threshold)
    and how many benign samples would be correctly fast-pathed at each.
    """
    if std_multipliers is None:
        std_multipliers = np.arange(0.0, 1.05, 0.1)

    model.eval()
    all_scores, all_labels = [], []

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

    attack_scores = scores[labels == 1]
    benign_scores = scores[labels == 0]

    score_mean_benign = float(benign_scores.mean())
    score_std_benign = float(benign_scores.std())

    rows = []
    for k in std_multipliers:
        candidate_low = score_mean_benign + k * score_std_benign
        attacks_bypassed = int((attack_scores < candidate_low).sum())
        benign_fast_pathed = int((benign_scores < candidate_low).sum())

        rows.append(
            {
                "std_multiplier": round(float(k), 2),
                "candidate_low_threshold": candidate_low,
                "attacks_bypassed": attacks_bypassed,
                "attacks_bypassed_pct": attacks_bypassed / len(attack_scores),
                "benign_fast_pathed": benign_fast_pathed,
                "benign_fast_pathed_pct": benign_fast_pathed / len(benign_scores),
            }
        )

    print(
        f"score_mean_benign={score_mean_benign:.6f}, score_std_benign={score_std_benign:.6f}"
    )
    return pl.DataFrame(rows)


if __name__ == "__main__":
    raw_dataset = DATA_RAW_DIR / "shieldlm_ood_dataset.parquet"

    processed_dataset = DATA_PROCESSED_DIR / "shield_ood_dataset.npz"

    scaler_file = DATA_PROCESSED_DIR / "SCALER_FILE_FOR_FULL_SEED_3123.pt"

    ood_data.build_parquet_file(raw_dataset)

    struct_fn = StructuralExtractor()

    if not processed_dataset.exists():
        paths = ood_data.process_data(
            raw_dataset, processed_dataset, structural_fn=struct_fn, save_n=1000
        )

        merge_parts_simple(paths, out_path=processed_dataset)

    dataset = PromptFeatureDataset(processed_dataset)

    scaler = StandardScaler().load(scaler_file)

    dataset.features = scaler.transform(dataset.features)

    dataloader = DataLoader(dataset=dataset, batch_size=512, shuffle=False)

    args = Args(ae_bottleneck=100, expert_k=4)

    base_model = BaseNormalAutoEncoder(len(dataset[0][0]), args=args, with_dropout=True)
    normal_model = NormalityAE(
        input_dim=len(dataset[0][0]), args=args, _mode_2=True, with_dropout=True
    )

    base_weights = torch.load(
        MODEL_DIRECTORY_DEV / "BaseNormalAutoEncoder_instance_1789824192.1090372"
    )
    normality_weights = torch.load(
        MODEL_DIRECTORY_DEV / "NormalityAE_instance_1789824349.7443838"
    )

    base_model.load_state_dict(base_weights["model"])
    print(base_weights["epoch"])

    normal_model.load_state_dict(normality_weights["model"])
    print(normality_weights["epoch"])
    base_model_results = ood_data.evaluate_ood_metrics(
        model=base_model,
        loader=dataloader,
        model_name=base_model.__class__.__name__,
        categories=dataset.extra.get("label_category"),
    )

    print("BASE OOD Results: ")
    base_model_results.select("model_name", "roc_auc", "auprc").show(None)

    base_model_results.write_csv(TRAINING_LOGS_DIR / "base_model_results.csv")
    normal_model_results = ood_data.evaluate_ood_metrics(
        model=normal_model,
        loader=dataloader,
        model_name=normal_model.__class__.__name__,
        categories=dataset.extra.get("label_category"),
    )

    print("Normality OOD Results: ")
    print(normal_model_results.select("model_name", "roc_auc", "auprc").show(None))

    normal_model_results.write_csv(TRAINING_LOGS_DIR / "normality_results.csv")

    idd_dataset = PromptFeatureDataset(
        DATA_PROCESSED_DIR / "final_data_processed_structural_3123-nomic-embed-text.npz"
    )

    idd_dataset.features = scaler.transform(idd_dataset.features)

    split_sampler = SplitSampler.load_file(
        DATA_PROCESSED_DIR
        / "SAMPLER_final_data_seed_processed_3123_TRAIN_RATIO_0.7_TEST_RATIO_0.15.pt"
    )

    val_dataset = Subset(
        idd_dataset, idd_dataset.ids_to_positions(split_sampler.get_split("validation"))
    )

    val_loader = DataLoader(val_dataset, batch_size=512)
    categories = dataset.extra.get("label_category")

    full_table_base = evaluate_model_with_all_thresholds(
        model=base_model,
        val_loader=val_loader,
        dataloader=dataloader,
        categories=categories,
        out_path=TRAINING_LOGS_DIR / "base_model_results_with_threshold.csv",
    )

    full_table_normal = evaluate_model_with_all_thresholds(
        model=normal_model,
        val_loader=val_loader,
        dataloader=dataloader,
        categories=categories,
        out_path=TRAINING_LOGS_DIR / "normal_model_results_with_threshold.csv",
    )

    ood_sampler = SplitSampler(
        train_split=0.0, test_split=0.8, validation_split=0.2, seed=3123
    )

    ood_sampler.build_split(dataset.item_ids, dataset.labels)

    val_ids = ood_sampler.get_split("validation")
    test_ids = ood_sampler.get_split("test")

    val_positions = dataset.ids_to_positions(val_ids)
    test_positions = dataset.ids_to_positions(test_ids)

    full_categories = dataset.extra.get(
        "label_category"
    )  # full-length array, aligned to `dataset`

    val_categories = full_categories[val_positions]
    test_categories = full_categories[test_positions]

    val_ood_dataset = Subset(dataset, val_positions)
    test_ood_dataset = Subset(dataset, test_positions)

    val_ood_loader = DataLoader(val_ood_dataset, batch_size=512)
    test_ood_loader = DataLoader(test_ood_dataset, batch_size=512)
    full_table_base = evaluate_model_with_all_thresholds(
        model=base_model,
        val_loader=val_ood_loader,
        dataloader=test_ood_loader,
        categories=test_categories,
        out_path=TRAINING_LOGS_DIR / "base_model_results_with_ood_threshold.csv",
    )

    full_table_normal = evaluate_model_with_all_thresholds(
        model=normal_model,
        val_loader=val_ood_loader,
        dataloader=test_ood_loader,
        categories=test_categories,
        out_path=TRAINING_LOGS_DIR / "normal_model_results_with_ood_threshold.csv",
    )
