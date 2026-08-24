import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from ml_factory.datasets import PromptFeatureDataset
from ml_factory.models.autoencoder import NormalityResult
from ml_factory.utils.scaler import StandardScaler
from ml_factory.datasets.sampler import SplitSampler
from ml_factory.models import Args, BaseNormalAutoEncoder, NormalityAE
from ml_factory import MODEL_DIRECTORY_DEV, DATA_PROCESSED_DIR, DATA_RAW_DIR
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report, accuracy_score, balanced_accuracy_score, f1_score, auc, roc_curve
from typing import List
import numpy as np
import polars as pl
import torch.nn as nn
from ml_factory.training_scripts.scripts_ae import TEST_RATIO


EMBEDDING_MODEL = 'nomic-embed-text'
SEED = 3123
TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



def plot_threshold_f1(
    model: nn.Module,
    model_files: dict[str, str],
    val_loader: DataLoader,
    device: str = "cuda",
    n_thresholds: int = 200,
):
    model.to(device)
    model.eval()

    plt.figure(figsize=(9, 6))

    for variant, file_name in model_files.items():

        checkpoint = torch.load(
            MODEL_DIRECTORY_DEV / file_name,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model"])
        model.eval()

        y_true = []
        scores = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)

                result = model(X)

                if isinstance(result, tuple):
                    recon = result[0]
                else:
                    recon = result.recon

                error = torch.mean(
                    (recon - X) ** 2,
                    dim=1
                )

                y_true.extend(y.cpu().numpy())
                scores.extend(error.cpu().numpy())

        y_true = np.asarray(y_true)
        scores = np.asarray(scores)

        thresholds = np.linspace(
            scores.min(),
            scores.max(),
            n_thresholds
        )

        f1_scores = []

        for threshold in thresholds:

            y_pred = (scores > threshold).astype(int)

            f1 = f1_score(
                y_true,
                y_pred,
                pos_label=1
            )

            f1_scores.append(f1)

        f1_scores = np.asarray(f1_scores)

        best_idx = np.argmax(f1_scores)

        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        plt.plot(
            thresholds,
            f1_scores,
            label=(
                f"{variant} "
                f"(τ={best_threshold:.3f}, "
                f"F1={best_f1:.3f})"
            )
        )

        plt.scatter(
            best_threshold,
            best_f1,
            s=50
        )

    plt.xlabel("Reconstruction Error Threshold")
    plt.ylabel("Attack F1")
    plt.title(f"{model.__class__.__name__} Threshold Selection on Validation Set")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model.__class__.__name__}_Threshold_Selection.png')
    plt.show()

def evaluate_models(
    model,
    model_files: dict[str, str],
    thresholds: dict[str, float],
    test_loader,
    device: str = "cuda",
) -> pl.DataFrame:

    rows = []

    for variant, file_name in model_files.items():

        # Load model
        checkpoint = torch.load(MODEL_DIRECTORY_DEV / file_name, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()

        y_true = []
        scores = []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)

                result = model(X)

                # Reconstruction error for each sample
                if isinstance(result, NormalityResult):

                    errors = torch.mean(
                        (result.recon - X) ** 2,
                        dim=1
                    )
                else:
                    errors = torch.mean(
                        (result[0] - X) ** 2,
                        dim=1
                    )

                y_true.extend(y.cpu().numpy())
                scores.extend(errors.cpu().numpy())

        y_true = np.asarray(y_true)
        scores = np.asarray(scores)

        threshold = thresholds[variant]

        # Reconstruction error > threshold => attack
        y_pred = (scores > threshold).astype(int)

        rows.append({
            "Variant": variant,
            "Threshold": threshold,
            "ROC-AUC": roc_auc_score(y_true, scores),
            "Avg. Precision": average_precision_score(y_true, scores),
            "Attack F1": f1_score(y_true, y_pred, pos_label=1),
            "Balanced Acc.": balanced_accuracy_score(y_true, y_pred),
        })

    return pl.DataFrame(rows)

def find_optimal_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    models: dict[str, str],
    device: str = DEVICE
) -> dict[str, float]:

    evaluation = {}

    for model_type, file_name in models.items():

        # Load model weights
        model.load_state_dict(
            torch.load(
                MODEL_DIRECTORY_DEV / file_name,
                map_location=device
            )['model']
        )

        model.to(device)
        model.eval()

        scores = []
        labels = []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)

                result = model(X)

                # Reconstruction error = anomaly score
                if isinstance(result, NormalityResult):
                    reconstruction_error = (
                        (result.recon - X) ** 2
                    ).mean(dim=1)
                else:
                    reconstruction_error = (
                        (result[0] - X) ** 2
                    ).mean(dim=1)

                scores.append(reconstruction_error.cpu())
                labels.append(y.cpu())

        scores = torch.cat(scores).numpy()
        labels = torch.cat(labels).numpy()

        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(
            labels,
            scores
        )

        # F1 for each threshold
        f1 = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )

        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]

        evaluation[model_type] = float(best_threshold)

        print(
            f"{model_type:<25} "
            f"threshold={best_threshold:.6f} "
            f"val_f1={f1[best_idx]:.4f}"
        )

    return evaluation



def plot_auc_pr_curves(
    model: nn.Module,
    model_files: dict[str, str],
    test_loader: DataLoader,
    device: str = "cuda",
):
    model.to(device)
    model.eval()

    plt.figure(figsize=(8, 6))

    for variant, file_name in model_files.items():

        checkpoint = torch.load(
            MODEL_DIRECTORY_DEV/ file_name,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model"])
        model.eval()

        y_true = []
        scores = []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)

                result = model(X)

                if isinstance(result, tuple):
                    recon = result[0]
                else:
                    recon = result.recon

                # One reconstruction error per sample
                error = torch.mean(
                    (recon - X) ** 2,
                    dim=1
                )

                y_true.extend(y.cpu().numpy())
                scores.extend(error.cpu().numpy())

        y_true = np.asarray(y_true)
        scores = np.asarray(scores)

        # -------------------------
        # ROC
        # -------------------------
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{variant} (AUC = {roc_auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model.__class__.__name__} ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model.__class__.__name__}_ROC_AUC_CURVE.png')
    plt.show()

    # ------------------------------------------------
    # PR CURVES
    # ------------------------------------------------

    plt.figure(figsize=(8, 6))

    for variant, file_name in model_files.items():

        checkpoint = torch.load(
            MODEL_DIRECTORY_DEV / file_name,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model"])
        model.eval()

        y_true = []
        scores = []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)

                result = model(X)

                if isinstance(result, tuple):
                    recon = result[0]
                else:
                    recon = result.recon

                error = torch.mean(
                    (recon - X) ** 2,
                    dim=1
                )

                y_true.extend(y.cpu().numpy())
                scores.extend(error.cpu().numpy())

        y_true = np.asarray(y_true)
        scores = np.asarray(scores)

        precision, recall, _ = precision_recall_curve(
            y_true,
            scores
        )

        ap = average_precision_score(
            y_true,
            scores
        )

        plt.plot(
            recall,
            precision,
            label=f"{variant} (AP = {ap:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model.__class__.__name__} Precision-Recall Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model.__class__.__name__}_PR_AUC_CURVE.png')
    plt.show()


if __name__ == '__main__':
    normality_ae_models = {
            'with_diversity_router': 'NormalityAE_attack_router_1787375116.7284417.pt',
            'with_diversity': 'NormalityAE_attack_1787373164.6574042.pt',
            'with_attack': 'NormalityAE_attack_1787368973.9307342.pt',
            'base': 'NormalityAE_1787367650.579661.pt'
            }

    base_ae_models = {
            'with_contrastive': 'BaseNormalAutoEncoder_attack_1787366956.0810184.pt',
            'with_attack': 'BaseNormalAutoEncoder_attack_1787366425.8858726.pt',
            'base': 'BaseNormalAutoEncoder_1787365906.2026029.pt'
            }


    processed_data = DATA_PROCESSED_DIR / f'processed_final_data_{EMBEDDING_MODEL}.npz'

    if not processed_data.is_file():
        raise FileNotFoundError(f'file for testing not found: {processed_data}')


    dataset = PromptFeatureDataset(processed_data)

    split_path = DATA_PROCESSED_DIR / f'processed_final_data_seed_{SEED}_train_{TRAIN_RATIO}_test_{TEST_RATIO}.pt'

    if not split_path.is_file():
        splitter = SplitSampler(train_split=TRAIN_RATIO, test_split=TEST_RATIO, validation_split=VALIDATION_RATIO, seed=SEED)
        splitter.build_split(item_ids=dataset.item_ids, labels=dataset.labels)
        splitter.save_split(split_path)

    else:
        splitter = SplitSampler.load_file(split_path)


    std_scaler_path = DATA_PROCESSED_DIR / f'processed_final_data_seed_{SEED}_train_{TRAIN_RATIO}_standard_sclaer.pt'

    scaler = StandardScaler.load(std_scaler_path)

    dataset.features = scaler.transform(dataset.features)

    test_data = Subset(dataset=dataset, indices=dataset.ids_to_positions(splitter.get_split('test')))

    validation_data = Subset(dataset=dataset, indices=dataset.ids_to_positions(splitter.get_split('validation')))

    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=128,shuffle=False, num_workers=0)

    args = Args()

    context_window = len(dataset[0][0])
    print(context_window)
    base_model = BaseNormalAutoEncoder(context_window, args).to(DEVICE)

    normality_model = NormalityAE(context_window, args).to(DEVICE)

    base_model.eval()

    normality_model.eval()

    print('Thresholds for BaseNormalAutoEncoder')
    base_ae_thresholds = find_optimal_threshold(
        base_model,
        validation_loader,
        base_ae_models
    )

    print(base_ae_thresholds)

    print('Thresholds for NormalityAE')
    normality_thresholds = find_optimal_threshold(
        normality_model,
        validation_loader,
        normality_ae_models
    )

    print(normality_thresholds)


    print('\n' + '#' * 70)
    print('BASE NORMAL AUTOENCODER - TEST RESULTS')
    print('#' * 70)

    evaluation_base_ae = evaluate_models(
        model=base_model,
        model_files=base_ae_models,
        thresholds=base_ae_thresholds,
        test_loader=test_loader
    )
    
    print(evaluation_base_ae.show(ascii_tables=True))

    print('\n' + '#' * 70)
    print('NORMALITY AE - TEST RESULTS')
    print('#' * 70)

    evaluation_normality_ae = evaluate_models(
        model=normality_model,
        model_files=normality_ae_models,
        thresholds=normality_thresholds,
        test_loader=test_loader
    )

    print(evaluation_normality_ae.show(ascii_tables=True))
    plot_auc_pr_curves(
        base_model,
        base_ae_models,
        test_loader,
        DEVICE
    )

    plot_auc_pr_curves(
        normality_model,
        normality_ae_models,
        test_loader,
        DEVICE
    )

    plot_threshold_f1(
    base_model,
    base_ae_models,
    validation_loader,
    DEVICE)

    plot_threshold_f1(
        normality_model,
        normality_ae_models,
        validation_loader,
        DEVICE
    )

