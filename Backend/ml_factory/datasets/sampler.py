import hashlib
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Literal, Optional, Self
from collections.abc import Iterable

import polars as pl
import torch
from torch.utils.data import Sampler

data = pl.DataFrame()
TOLERANCE = 1e-6


class RatioSampler(Sampler):
    def __init__(
        self, labels, attack_label=1, attack_ratio=0.1, seed=100, epoch_length=None
    ):
        labels = torch.as_tensor(labels)
        self.benign_idx = torch.where(labels != attack_label)[0]
        self.attack_idx = torch.where(labels == attack_label)[0]

        if len(self.benign_idx) == 0 or len(self.attack_idx) == 0:
            raise ValueError("Both benign and attack pools must be non empty")

        self.attack_ratio = attack_ratio
        self.seed = seed
        self.epoch = 0

        self.epoch_length = epoch_length or len(self.benign_idx)
        self.n_attack = int(round(self.epoch_length * attack_ratio))
        self.n_benign = self.epoch_length - self.n_attack

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()

        g.manual_seed(self.seed + self.epoch)

        if self.n_benign <= len(self.benign_idx):
            perm = torch.randperm(len(self.benign_idx), generator=g)[: self.n_benign]
            benign_sample = self.benign_idx[perm]

        else:
            sel = torch.randint(0, len(self.benign_idx), (self.n_benign,), generator=g)
            benign_sample = self.benign_idx[sel]

        if self.n_attack <= len(self.attack_idx):
            perm = torch.randperm(len(self.attack_idx), generator=g)[: self.n_attack]
            attack_sample = self.attack_idx[perm]

        else:
            sel = torch.randint(0, len(self.attack_idx), (self.n_attack,), generator=g)
            attack_sample = self.attack_idx[sel]

        combined = torch.cat([benign_sample, attack_sample])
        shuffle = torch.randperm(len(combined), generator=g)

        return iter(combined[shuffle].tolist())

    def __len__(self):
        return self.epoch_length


class SplitSampler:
    def __init__(
        self,
        train_split: float = 0.7,
        test_split: float = 0.15,
        validation_split: float = 0.15,
        seed: int = 432,
    ):
        abs_split_error = abs(1 - (train_split + test_split + validation_split))
        if abs_split_error > TOLERANCE:
            raise ValueError(
                f"train test and validation split must equal to one, currently got: {train_split=}, {test_split=}, {validation_split=}"
            )

        self._train_split = train_split
        self._test_split = test_split
        self._validation_split = validation_split
        self.__seed = seed

        self.__split: Optional[dict] = None

    def get_split(self, key: Literal["train", "test", "validation"]):
        if self.__split is None:
            raise ValueError("SplitSampler Not Built...")
        return self.__split[key]

    def __set_split(self, value):
        self.__split = value

    @classmethod
    def load_file(cls, path: Path | PathLike) -> Self:

        loaded = torch.load(path)

        loaded_sampler = cls(
            train_split=loaded["train_split"],
            test_split=loaded["test_split"],
            validation_split=loaded["validation_split"],
            seed=loaded["seed"],
        )

        loaded_sampler.__set_split(loaded["split"])

        return loaded_sampler

    def build_split(self, item_ids: Iterable, labels: Iterable) -> Self:
        if self.__split is not None:
            raise Exception("Split is already built")

        by_label = defaultdict(list)

        seen: dict = {}

        for item_id, label in zip(item_ids, labels):
            if item_id in seen and seen[item_id] != label:
                raise ValueError(
                    f"Inconsistent Label for Item ID= {item_id!r}, saw {seen[item_id]!r} and {label!r}"
                )

            seen[item_id] = label

        for pos, (item_id, label) in enumerate(zip(item_ids, labels)):
            by_label[label].append((pos, item_id))

        splits = {"train": [], "validation": [], "test": []}

        for item_id, label in seen.items():
            h = int(
                hashlib.sha256(f"{self.__seed}-{label}-{item_id}".encode()).hexdigest(),
                16,
            )

            frac = (h % 10_000) / 10_000
            if frac < self._test_split:
                splits["test"].append(item_id)

            elif frac < self._test_split + self._validation_split:
                splits["validation"].append(item_id)

            else:
                splits["train"].append(item_id)

        self.__set_split(splits)
        return self

    def save_split(self, path: Path | PathLike) -> None:

        torch.save(
            {
                "train_split": self._train_split,
                "test_split": self._test_split,
                "validation_split": self._validation_split,
                "split": self.__split,
                "seed": self.__seed,
            },
            path,
        )
