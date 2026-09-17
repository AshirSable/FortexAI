import inspect

from sklearn.externals.array_api_compat.numpy import average

from ml_factory.evaluation.abstract import EvaluationEngine
from typing import Callable, Literal, Optional, List
import numpy as np

import sklearn.metrics as pmeter


def metrics_ctx_helper(
    y_pred: Optional[List[float] | np.ndarray] = None,
    y_actual: Optional[List[float] | np.ndarray] = None,
    y_score: Optional[List[float] | np.ndarray] = None,
) -> dict[str, Optional[List[float] | np.ndarray]]:
    return {"y_pred": y_pred, "y_actual": y_actual, "y_score": y_score}


def accuracy_score(y_pred, y_actual, **_):
    return pmeter.accuracy_score(y_actual, y_pred)


def auprc(
    y_score,
    y_actual,
    average: Literal["micro", "samples", "weighted", "macro"] = "macro",
    **_,
):
    return pmeter.average_precision_score(y_actual, y_score, average=average)


def f1_score(y_pred, y_actual, **_):
    return pmeter.f1_score(
        y_actual,
        y_pred,
        pos_label=_["pos_label"] or 1,
        average=_.get("average") or "binary",
    )


def recall(y_pred, y_actual, **_):
    return pmeter.recall_score(
        y_actual,
        y_pred,
        pos_label=_["pos_label"] or 1,
        average=_.get("average") or "binary",
    )


def precision(y_pred, y_actual, **_):
    return pmeter.precision_score(
        y_actual,
        y_pred,
        pos_label=_["pos_label"] or 1,
        average=_.get("average") or "binary",
    )


def roc_auc(y_score, y_actual, **_):
    return pmeter.roc_auc_score(y_actual, y_score, average=_.get("average") or "macro")


class MetricsEngine(EvaluationEngine):
    def __init__(self, metrics_map: dict[str, Callable], include_metrics: list):

        self.metrics_map = metrics_map

        self.include_metrics = include_metrics

        self.__params_cache = {
            name: set(inspect.signature(fn).parameters.keys())
            for name, fn in metrics_map.items()
        }

    def compute(
        self,
        ctx: dict[
            Literal["y_pred", "y_actual", "y_score"], Optional[List[float] | np.ndarray]
        ],
    ) -> dict[str, float]:
        # THOUGHT: what will the ctx contain?
        # maybe information on y_pred and y_actual and y_score would be whats needed
        # so make a dictionary that would contain it..

        individual_metrics = {}

        for metrics in self.include_metrics:
            metric_fn = self.metrics_map[metrics]
            fn_params = self.__params_cache[metrics]

            filtered_kwargs = {k: v for k, v in ctx.items() if k in fn_params}

            metric_val = metric_fn(**filtered_kwargs)
            individual_metrics[metrics] = metric_val

        return individual_metrics


from ml_factory.evaluation import (
    ACCURACY_METRIC,
    AUPRC_METRIC,
    F1_METRIC,
    PRECISION_METRIC,
    RECALL_METRIC,
    ROCAUC_METRIC,
)

METRICS_MAP = {
    AUPRC_METRIC: auprc,
    ACCURACY_METRIC: accuracy_score,
    F1_METRIC: f1_score,
    PRECISION_METRIC: precision,
    RECALL_METRIC: recall,
    ROCAUC_METRIC: roc_auc,
}
