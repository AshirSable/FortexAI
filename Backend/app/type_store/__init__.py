from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, NoReturn, TypeAlias, TypeVar

import numpy as np
import torch

from app.type_store._error import InputValidationError

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T]):
    _value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self._value

    def unwrap_or(self, default: T) -> T:
        return self._value


@dataclass(frozen=True)
class Err(Generic[E]):
    _error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> NoReturn:
        raise ValueError(f"Called Unwrap on an Err value: {self._error}")

    def unwrap_or(self, default: T) -> T:
        return default


Result: TypeAlias = Ok[T] | Err[E]


@dataclass(frozen=True)
class PhaseInput:
    text: str | None = None
    embedding: torch.Tensor | np.ndarray | None = None
    _prior_results: dict[Phase, SuccessReturn] = field(default_factory=dict)

    def require_text(self) -> str:
        if self.text == None:
            raise InputValidationError("This phase requires `text`, got None")

        return self.text

    def require_embedding(self) -> torch.Tensor | np.ndarray:
        if self.embedding == None:
            raise InputValidationError("This phase requires `embedding`, got None")

        return self.embedding


class Verdict(int, Enum):
    benign = 0
    attack = 1


class Phase(int, Enum):
    semantic_search = 100
    autoencoder = 101
    ensemble_bert = 102
    llm_judge = 103


@dataclass
class SuccessReturn:
    verdict: Verdict
    at_phase: Phase
    confidence: float


@dataclass
class SuccessForReview:
    success_return: SuccessReturn
    embedding: torch.Tensor | np.ndarray
    text: str
