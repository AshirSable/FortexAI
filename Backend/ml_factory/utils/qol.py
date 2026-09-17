import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from datasets.utils.py_utils import Literal
from torch import Size
from torch.utils.data import DataLoader, Subset

from ml_factory.datasets import (
    PromptFeatureDataset,
)
from ml_factory.datasets.precompute_features import precompute_features_ae
from ml_factory.datasets.sampler import RatioSampler, SplitSampler
from ml_factory.utils import embedding_text, give_id_to_data, merge_parts
from ml_factory.utils.scaler import StandardScaler
from ml_factory.utils.structural_extractor import StructuralExtractor


@dataclass
class ResultLoaders:
    train: DataLoader | None
    validation: DataLoader | None
    test: DataLoader | None
    x_dim: Size
    y_dim: Size
    _len: int


def load_dataset_template(
    data_file: Path | None = None,
    raw_data_file: Path | None = None,
    sampler_file: Path | None = None,
    scaler_file: Path | None = None,
    ratio_sampler: bool = False,
    TRAIN_RATIO: float = 0.7,
    TEST_RATIO: float = 0.15,
    VALIDATION_RATIO: float = 0.15,
    SEED: int = 3123,
    EMBEDDING_MODEL: str = "nomic-embed-text",
    embedding_fn: Callable[..., Awaitable[list]] = embedding_text,
    structural_exec: StructuralExtractor | None = None,
    ATTACK_RATIO: float = 0.2,
    epoch_length: int | None = None,
    BATCH_SIZE: int = 128,
    save_n: Literal["all"] | int = 1000,
):

    if data_file is None or not data_file.is_file():
        print(data_file, "given data file")
        print(data_file.is_file(), "given data file")
        assert (
            raw_data_file is not None
        ), "Need to Provide raw_data_file if data_file is not given or found"
        df = pl.scan_parquet(raw_data_file)

        df = give_id_to_data(df)

        paths = asyncio.run(
            precompute_features_ae(
                dataset_records=df,
                embed_fn=embedding_fn,
                embedding_model_name=EMBEDDING_MODEL,
                out_path=data_file or Path("processed_file.npz"),
                batch_size=1000,
                chunk_size=50,
                max_concurrent=os.cpu_count() or 6,
                structural_extractor=structural_exec,
                save_n=save_n,
            )
        )

        if save_n != "all":
            merge_parts(paths, data_file or Path("processed_file.npz"))

    dataset = PromptFeatureDataset(data_file)

    if sampler_file is None or not sampler_file.is_file():
        splitter = SplitSampler(
            train_split=TRAIN_RATIO,
            test_split=TEST_RATIO,
            validation_split=VALIDATION_RATIO,
            seed=SEED,
        )

        splitter.build_split(item_ids=dataset.item_ids, labels=dataset.labels)
        if sampler_file is not None:
            splitter.save_split(sampler_file)

    else:
        splitter = SplitSampler.load_file(sampler_file)

    train_ids = splitter.get_split("train")
    test_ids = splitter.get_split("test")
    validation_ids = splitter.get_split("validation")

    train_positions = dataset.ids_to_positions(train_ids)
    if scaler_file is None or not scaler_file.is_file():
        scaler = StandardScaler()
        scaler = scaler.fit(dataset.features[train_positions])
        dataset.features = scaler.transform(dataset.features)

        if scaler_file is not None:
            scaler.save(scaler_file)

    else:
        scaler = StandardScaler.load(scaler_file)
        dataset.features = scaler.transform(dataset.features)

    train_data = Subset(dataset=dataset, indices=train_positions)
    validation_data = Subset(
        dataset=dataset, indices=dataset.ids_to_positions(validation_ids)
    )
    test_data = Subset(dataset=dataset, indices=dataset.ids_to_positions(test_ids))

    rtio_splr = None
    if ratio_sampler:
        rtio_splr = RatioSampler(
            labels=dataset.labels[train_positions],
            epoch_length=epoch_length,
            attack_label=1,
            attack_ratio=ATTACK_RATIO,
            seed=SEED,
        )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=not ratio_sampler,
        sampler=rtio_splr,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return ResultLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        x_dim=dataset[0][0].shape,
        y_dim=dataset[0][1].shape,
        _len=len(dataset),
    )
