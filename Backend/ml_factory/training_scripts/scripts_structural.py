import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
from nltk import Path

from ml_factory import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    MODEL_DIRECTORY_DEV,
    TRAINING_LOGS_DIR,
)
from ml_factory.training_scripts.training_script_ae_structural import (
    training_attack_malleable,
)
from ml_factory.utils import (
    CONFIG,
    ModelSaver,
    Tracker,
    embedding_text,
)
from ml_factory.utils.plotter import plotting_logs
from ml_factory.utils.qol import ResultLoaders, load_dataset_template
from ml_factory.utils.structural_extractor import StructuralExtractor
from ml_factory.utils.trainer import TrainerArgs


@dataclass
class Files:
    raw_data_file: Path
    processed_data_file: Path
    sampler_file: Path
    structural_extractor_file: Path
    scaler_file: Path


def get_files_for_benign(config: CONFIG):
    return Files(
        raw_data_file=DATA_RAW_DIR / "only_benign_prompts.parquet",
        processed_data_file=(
            DATA_PROCESSED_DIR
            / f"only_benign_prompts_processed_structural_{config.seed}-{config.embedding_model}.npz"
        ),
        sampler_file=DATA_PROCESSED_DIR
        / f"SAMPLER_only_benign_prompts_seed_{config.seed}_TRAIN_RATIO_{config.train_ratio}_TEST_RATIO_{config.test_ratio}.pt",
        structural_extractor_file=DATA_PROCESSED_DIR / f"structural_extractor.json",
        scaler_file=DATA_PROCESSED_DIR
        / f"SCALER_FILE_FOR_BENIGN_SEED_{config.seed}.pt",
    )


def get_files_for_full_data(config: CONFIG):
    return Files(
        raw_data_file=DATA_RAW_DIR / "final_data.parquet",
        processed_data_file=(
            DATA_PROCESSED_DIR
            / f"final_data_processed_structural_{config.seed}-{config.embedding_model}.npz"
        ),
        sampler_file=DATA_PROCESSED_DIR
        / f"SAMPLER_final_data_seed_processed_{config.seed}_TRAIN_RATIO_{config.train_ratio}_TEST_RATIO_{config.test_ratio}.pt",
        structural_extractor_file=DATA_PROCESSED_DIR / f"structural_extractor.json",
        scaler_file=DATA_PROCESSED_DIR / f"SCALER_FILE_FOR_FULL_SEED_{config.seed}.pt",
    )


def get_benign_data(config: CONFIG, files: Optional[Files] = None):
    files = files or get_files_for_benign(config)

    if files.structural_extractor_file.is_file():
        structural_extractor = StructuralExtractor.from_json(
            files.structural_extractor_file
        )
    else:
        structural_extractor = StructuralExtractor()
        structural_extractor.save_json(files.structural_extractor_file)

    benign_data_loaders = load_dataset_template(
        data_file=files.processed_data_file,
        raw_data_file=files.raw_data_file,
        sampler_file=files.sampler_file,
        ratio_sampler=config.ratio_sampler,
        TRAIN_RATIO=config.train_ratio,
        TEST_RATIO=config.test_ratio,
        VALIDATION_RATIO=config.validation_ratio,
        SEED=config.seed,
        EMBEDDING_MODEL=config.embedding_model,
        embedding_fn=embedding_text,
        structural_exec=structural_extractor,
        BATCH_SIZE=config.batch_size,
    )

    return benign_data_loaders


def get_full_data(config: CONFIG, files: Optional[Files] = None):
    files = files or get_files_for_full_data(config)

    if files.structural_extractor_file.is_file():
        structural_extractor = StructuralExtractor.from_json(
            files.structural_extractor_file
        )
    else:
        structural_extractor = StructuralExtractor()
        structural_extractor.save_json(files.structural_extractor_file)

    full_data_loaders = load_dataset_template(
        scaler_file=files.scaler_file,
        data_file=files.processed_data_file,
        raw_data_file=files.raw_data_file,
        sampler_file=files.sampler_file,
        ratio_sampler=config.ratio_sampler,
        TRAIN_RATIO=config.train_ratio,
        TEST_RATIO=config.test_ratio,
        VALIDATION_RATIO=config.validation_ratio,
        SEED=config.seed,
        EMBEDDING_MODEL=config.embedding_model,
        embedding_fn=embedding_text,
        structural_exec=structural_extractor,
        BATCH_SIZE=config.batch_size,
    )

    return full_data_loaders


def config_print(config: CONFIG):
    print(
        "DETAILS: ",
        "\n",
        "loss functions used: ",
        ", ".join(config.include_loss.keys()),
    )
    print(
        "metrics used: ",
        (
            ", ".join(config.include_metrics)
            if len(config.include_metrics) != 0
            else "no metrics used"
        ),
    )
    print(
        "best model will be chosen with: ",
        f"{config.best_metric!r}",
        "with condition: ",
        f"{config.best_metric_condition!r}",
        "(le: <=, ge: >=)",
    )
    attack_data_cond = config.ratio_sampler and config.attack_ratio != 0.0
    print("contains attack data ?: ", "yes" if attack_data_cond else "no")
    if attack_data_cond:
        print("attack data ratio: ", config.attack_ratio)

    print("training epochs: ", config.epoch)

    print("training batch_size: ", config.batch_size)

    print("current seed: ", config.seed)

    print("current device: ", config.device)


def autoencoder_training_using(loaders: ResultLoaders, training_args: TrainerArgs):

    training_start = time.time()

    start_time = time.time()
    print(
        "#" * 20,
        "\n",
        datetime.now(),
        f"Training Started for MODEL: {training_args.model.__class__.__name__}, Instance {training_start}",
    )


def autoencoder_training(
    loaders: ResultLoaders,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: CONFIG,
    verbose: bool = True,
):
    start_time = time.time()
    print(
        "#" * 20,
        "\n",
        datetime.now(),
        f"Training Started for MODEL: {model.__class__.__name__}, Instance {start_time}",
    )
    if verbose:
        config_print(config)

    train_output = training_attack_malleable(
        train_loader=loaders.train,
        val_loader=loaders.validation,
        test_loader=loaders.test,
        model=model,
        optimizer=optimizer,
        epochs=config.epoch,
        device=config.device,
        loss_map=config.loss_map,
        include_loss=config.include_loss,
        include_metrics=config.include_metrics,
        metric_map=config.metrics_map,
        best_metric=config.best_metric,
        ratio_sampler=config.ratio_sampler,
        best_condition=config.best_metric_condition,
        verbose=verbose,
    )

    end_time = time.time() - start_time

    print(
        f"MODEL TRAINING: Instance {start_time} {model.__class__.__name__} Finished in {end_time} seconds"
    )

    training_logs: Tracker = train_output["train_log"]
    validation_logs: Tracker = train_output["val_log"]
    testing_logs: Tracker = train_output["test_log"]
    model_settings: ModelSaver = train_output["model"]

    torch.save(
        model_settings.get_dict(),
        MODEL_DIRECTORY_DEV / f"{model.__class__.__name__}_{start_time}.pt",
    )

    print(f"MODEL INSTANCE: {start_time} saved...")

    directory = TRAINING_LOGS_DIR / f"{model.__class__.__name__}_{start_time}"

    directory.mkdir(exist_ok=True)

    training_logs.to_csv(directory / f"train_log.csv")
    validation_logs.to_csv(directory / f"validation_log.csv")
    testing_logs.to_csv(directory / f"test_log.csv")

    if verbose:
        print(f"MODEL INSTANCE: {start_time} Saved Logs as CSV")

    plotting_logs(
        training_logs.logs,
        validation_logs.logs,
        path=directory / "plot_metrics_loss.png",
    )

    if verbose:
        print(f"MODEL INSTANCE: {start_time} Saved Plot")

    with open(directory / "parameters.json", "w") as f:
        json.dump(config.get_dict(), f)

    print("Training Finished", "\n\n", "#" * 20)


if __name__ == "__main__":
    SEED = 3123
    base_config = CONFIG(
        loss_map={},
        include_loss={},
        metrics_map={},
        include_metrics=[],
        best_metric="",
        best_metric_condition="ge",
        batch_size=128,
        seed=SEED,
        ratio_sampler=True,
        attack_ratio=0.2,
    )
    get_full_data(base_config)
