from abc import ABC, abstractmethod
from typing import Iterable


class EvaluationEngine(ABC):

    @abstractmethod
    def compute(self, ctx: dict) -> Iterable:
        raise NotImplementedError()
