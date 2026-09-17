from pathlib import Path
from typing import Optional, Self

import torch


class StandardScaler:
    def __init__(self, epsilon: float = 1e-6):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.epsilon: float = epsilon

    def __set_mean(self, mean: torch.Tensor):
        self.mean = mean

    def __set_std(self, std: torch.Tensor):
        self.std = std

    def fit(self, X: torch.Tensor) -> Self:

        mean = X.mean(dim=0)
        std = X.std(dim=0).clamp(min=self.epsilon)

        self.mean = mean
        self.std = std

        return self

    def save(self, path: Path | str):
        torch.save({"mean": self.mean, "std": self.std, "epsilon": self.epsilon}, path)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        loaded = torch.load(path)
        cl_ = cls(epsilon=loaded["epsilon"])

        cl_.__set_mean(loaded["mean"])
        cl_.__set_std(loaded["std"])

        return cl_

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "No Mean and Std Computed, please run .fit() method first"
            )
        return (X - self.mean) / self.std

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)

        return self.transform(X)
