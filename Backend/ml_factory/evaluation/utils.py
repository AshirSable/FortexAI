from functools import partial
from typing import Callable, List, Self


class EvaluationMapMapper:
    def __init__(self, map: dict[str, Callable]):
        self.__map = map
        self._kwargs_map = {}

        for k in map.keys():
            self._kwargs_map[k] = {}

    def add_kwargs(self, item: str, **kwargs) -> Self:
        self._kwargs_map[item].update(kwargs)

        self.__map[item] = partial(self.__map[item], **self._kwargs_map[item])

        return self

    def get_map(self):
        return self.__map

    def get_kwargs(self):

        return self.get_kwargs


class IncludeEvaluationMapper:
    def __init__(self, metrics: List[str]):
        self._metrics = metrics

        temp_map = {}

        for m in metrics:
            temp_map[m] = 1.0

        self.__map = temp_map

    def change_score(self, item, score: float) -> Self:
        if item not in self._metrics:
            raise KeyError(
                f" passed item {item!r} not found, available options: {self._metrics:!r}"
            )

        self.__map[item] = score

        return self

    def get_metrics_map(self):
        return self.__map
