import warnings
from collections import defaultdict
from pathlib import Path

import torch


class Tracker[T: (int, float, torch.Tensor)]:
    def __init__(self, *args: str, _counter_type: type[T] = torch.Tensor):
        self.logs = defaultdict(list)
        self.counter = defaultdict(_counter_type)
        self._counter_type: type[T] = _counter_type
        self.__make_dict(args)

    def _default_factory(self) -> T:
        if self._counter_type is torch.Tensor:
            return torch.tensor(0.0)

        return self._counter_type(0)

    def __get_value(self, value: T) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.item())
        else:
            return value

    def __make_dict(self, data: tuple[str, ...]):
        for d in data:
            self.logs[d] = []
            self.counter[d] = self._default_factory()

    def counter_reset(self):
        for key in self.counter:
            self.counter[key] = self._default_factory()

    def current_counter(self, key):
        return self.counter[key]

    def latest_log(self, key: str):
        return self.logs[key][-1]

    def counter_updates(self, data: dict):
        for k, v in data.items():
            self.counter_update(k, v)

    def _logs_update_map(self, data: dict[str, float], _by: float = 1):
        for key, value in data.items():
            self._logs_update_single(key=key, value=value, _by=_by)

    def _logs_update_single(self, key: str, value: float, _by: float = 1):
        if _by == 0:
            _by = 1
            warnings.warn("_by cannot be 0, keeping it as one")

        self.logs[key].append(value / _by)

    def counter_update(self, key: str, value: T):
        if isinstance(value, torch.Tensor):
            val_to_add = value.detach().cpu()
        else:
            val_to_add = value

        if self._counter_type is torch.Tensor:
            self.counter[key] += val_to_add
        else:
            self.counter[key] = self._counter_type(self.counter[key] + val_to_add)

    def update_logs(self, _by: int = 1, ignore: list[str] | None = None):
        ignore = ignore or []
        for key in self.counter:
            if not key in ignore:
                self._logs_update_single(
                    key, self.__get_value(self.counter[key]), _by=_by
                )

    def clear_logs(self):
        for key in self.logs:
            self.logs[key].clear()

    def to_csv(self, path: Path):
        import polars as pl

        pl.DataFrame(self.logs).write_csv(path)

    def to_parquet(self, path: Path):
        import polars as pl

        pl.DataFrame(self.logs).write_parquet(path)
