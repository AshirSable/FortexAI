from ml_factory.evaluation import TOTAL_LOSS
from enum import Enum


class EarlyStoppingEnum(Enum):
    stop = 1
    keep = 0


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        tolerance: float = 1e-6,
        _metric: str | None = None,
        _greater_is_better: bool = False,
    ):

        metric = _metric or TOTAL_LOSS
        self.patience = patience
        self.tolerance = tolerance

        # THOUGHT: What does early stopping actually perform
        # It checks whether for repeatedly if loss (val loss) keeps increasing or remains around same for patience epoch
        # in the end, EarlyStopping just needs to pass on a message for whether the loop should stop or not

        self._internal_clock = 0
        self._best_metric = float("-inf") if _greater_is_better else float("inf")
        self._greater_is_better = _greater_is_better
        self.metric = metric
        # THOUGHT: we will keep a track of the best loss found, if we find a loss less than the best loss we update our best loss with the new one and set the _internal_clock back to 0
        # ELSE what we do is keep increasing our internal clock, once it reaches or overbounds the patience parameter, we send out stop signal

    def __conditional_check(self, current_metric: float) -> bool:
        """Should give out True if current metric is worse then best metric"""
        if self._greater_is_better:
            return (self._best_metric + self.tolerance) >= current_metric
        else:
            return (self._best_metric - self.tolerance) <= current_metric

    def check(self, current_metric: float) -> EarlyStoppingEnum:

        if self.__conditional_check(current_metric):
            self._internal_clock += 1

        else:
            self._internal_clock = 0
            self._best_metric = current_metric

        if self._internal_clock >= self.patience:
            return EarlyStoppingEnum.stop
        return EarlyStoppingEnum.keep
