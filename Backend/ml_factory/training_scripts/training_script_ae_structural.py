from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Callable, Literal, Optional
import torch
import torch.nn as nn
from ml_factory.datasets.sampler import RatioSampler
from ml_factory.evaluation.metrics import METRICS_MAP, MetricsEngine, metrics_ctx_helper
from ml_factory.evaluation.loss import LOSS_MAP, LossEngine, create_ctx_context
from ml_factory.evaluation import (
    CONTRASTIVE_LOSS,
    DIVERSITY_LOSS,
    MSE_LOSS,
    OE_LOSS,
    ROUTER_LOAD_BALANCE_LOSS,
)
from ml_factory.models.autoencoder import ModelResult
from ml_factory.utils import Tracker, ModelSaver

LOSS_STR = "loss"
ATTACK_LOSS_STR = "attack_loss"
BENIGN_LOSS_STR = "benign_loss"


def training(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
    device: Literal["cpu", "cuda"] = "cuda",
):

    train_track = Tracker(LOSS_STR, _counter_type=torch.Tensor)
    validation_track = Tracker(LOSS_STR, _counter_type=torch.Tensor)

    criterion = nn.MSELoss()
    best_model = float("inf")
    best_model_settings = ModelSaver()

    for epoch in range(epochs):
        model.train()
        train_track.counter_reset()
        validation_track.counter_reset()
        for X, _ in train_loader:
            optimizer.zero_grad()
            X = X.to(device)
            result = model(X)
            loss = criterion(result.recon, X)
            loss.backward()
            optimizer.step()
            train_track.counter_update(key=LOSS_STR, value=loss.detach())

        model.eval()
        with torch.no_grad():
            for X, _ in val_loader:
                X = X.to(device)
                result = model(X)
                loss = criterion(result.recon, X)

                validation_track.counter_update(key=LOSS_STR, value=loss.detach())

        train_track.update_logs(_by=len(train_loader))
        validation_track.update_logs(_by=len(val_loader))

        if best_model > validation_track.current_counter(LOSS_STR):
            best_model_settings.save(model=model, optimizer=optimizer, epoch=epoch)

    test_tracker = Tracker(LOSS_STR)
    if test_loader is not None:
        with torch.no_grad():
            for X, _ in test_loader:
                X = X.to(device)
                result = model(X)
                loss = criterion(result.recon, X)

                test_tracker.counter_update(LOSS_STR, loss.detach())

            test_tracker.update_logs(_by=len(test_loader))

    return train_track.logs, validation_track.logs, test_tracker.logs


def condition_check(best_prev, now, condition="le"):
    if condition == "le":
        return best_prev <= now
    else:
        return best_prev >= now


def training_attack_malleable(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
    device: Literal["cpu", "cuda"] = "cuda",
    loss_map: dict[str, Callable] = LOSS_MAP,
    include_loss: dict[str, float] = {
        OE_LOSS: 1.0,
        DIVERSITY_LOSS: 1.0,
        CONTRASTIVE_LOSS: 1.0,
        ROUTER_LOAD_BALANCE_LOSS: 1.0,
        MSE_LOSS: 1.0,
    },
    metric_map: dict[str, Callable] = METRICS_MAP,
    include_metrics: list = [],
    metrics_on_training: bool = False,
    ratio_sampler: bool = False,
    best_metric: str = "auprc",
    best_condition: str = "le",
    verbose: bool = True,
):
    metrics_on_train_cond = metrics_on_training and len(include_metrics) != 0
    metrics_on_cond = len(include_metrics) != 0
    losses = list(include_loss.keys()) + [LOSS_STR]

    if metrics_on_cond:
        track_eval = losses + include_metrics
    else:
        track_eval = losses

    train_track = Tracker(
        *(track_eval if metrics_on_train_cond else losses), _counter_type=torch.Tensor
    )
    validation_track = Tracker(*track_eval, _counter_type=torch.Tensor)
    loss_engine = LossEngine(loss_map, include_loss)
    metric_engine = MetricsEngine(metric_map, include_metrics)
    best_model = float("inf")
    best_model_settings = ModelSaver()

    for epoch in range(epochs):
        model.train()
        train_track.counter_reset()
        validation_track.counter_reset()

        all_y_true = []
        all_y_score = []
        for X, y in train_loader:
            if ratio_sampler:
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)
            optimizer.zero_grad()
            y = y.to(device)
            X = X.to(device)
            result: ModelResult = model(X)
            ctx = create_ctx_context(X, y, result)
            total_loss, components = loss_engine.compute(ctx)
            total_loss.backward()
            optimizer.step()
            if metrics_on_train_cond:
                scores = ((result.recon - X) ** 2).mean(dim=1)
                all_y_score.append(scores.cpu())
                all_y_true.append(y.cpu())
            train_track.counter_updates(components)
            train_track.counter_update(LOSS_STR, total_loss.detach())

        train_track.update_logs(_by=len(train_loader), ignore=include_metrics)
        if metrics_on_train_cond:
            all_y_score = torch.cat(all_y_score).numpy()
            all_y_true = torch.cat(all_y_true).numpy()
            ctx = metrics_ctx_helper(y_actual=all_y_true, y_score=all_y_score)
            metrics_comp = metric_engine.compute(ctx)
            train_track._logs_update_map(metrics_comp, _by=1)

        all_y_true = []
        all_y_score = []
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                y = y.to(device)
                X = X.to(device)
                result: ModelResult = model(X)
                ctx = create_ctx_context(X, y, result)
                total_loss, components = loss_engine.compute(ctx)
                if metrics_on_cond:
                    scores = ((result.recon - X) ** 2).mean(dim=1)
                    all_y_score.append(scores.cpu())
                    all_y_true.append(y.cpu())

                validation_track.counter_updates(components)
                validation_track.counter_update(LOSS_STR, total_loss.detach())

        validation_track.update_logs(_by=len(val_loader), ignore=include_metrics)
        if metrics_on_cond:
            all_y_score = torch.cat(all_y_score).numpy()
            all_y_true = torch.cat(all_y_true).numpy()
            ctx = metrics_ctx_helper(y_actual=all_y_true, y_score=all_y_score)
            metrics_comp = metric_engine.compute(ctx)
            validation_track._logs_update_map(metrics_comp, _by=1)

        if condition_check(
            best_model, validation_track.latest_log(best_metric), best_condition
        ):
            best_model_settings.save(model=model, optimizer=optimizer, epoch=epoch)
            best_model = validation_track.latest_log(best_metric)
        if verbose:
            print(
                f"EPOCH [{epoch:<3}/{epochs}] | train loss: {train_track.latest_log(LOSS_STR):<5.5f}"
                f" | val loss: {validation_track.latest_log(LOSS_STR):<5.5f} | metric: {best_metric!r} {validation_track.latest_log(best_metric):<5.5f}"
            )

    test_tracker = Tracker(*track_eval, _counter_type=torch.Tensor)
    if test_loader is not None:
        with torch.no_grad():
            all_y_true = []
            all_y_score = []
            for X, y in test_loader:
                X = X.to(device)
                result = model(X)
                ctx = create_ctx_context(X, y, result)
                total_loss, components = loss_engine.compute(ctx)

                if metrics_on_cond:
                    scores = ((result.recon - X) ** 2).mean(dim=1)
                    all_y_score.append(scores.cpu())
                    all_y_true.append(y.cpu())
                test_tracker.counter_update(LOSS_STR, total_loss.detach())
                test_tracker.counter_updates(components)

            test_tracker.update_logs(_by=len(test_loader), ignore=include_metrics)
            if metrics_on_cond:
                all_y_score = torch.cat(all_y_score).numpy()
                all_y_true = torch.cat(all_y_true).numpy()
                ctx = metrics_ctx_helper(y_actual=all_y_true, y_score=all_y_score)
                metrics_comp = metric_engine.compute(ctx)
                test_tracker._logs_update_map(metrics_comp, _by=1)

    return {
        "train_log": train_track,
        "val_log": validation_track,
        "test_log": test_tracker,
        "model": best_model_settings,
    }
