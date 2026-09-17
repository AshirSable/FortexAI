from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_factory.evaluation.loss import LossEngine, create_ctx_context
from ml_factory.evaluation.metrics import MetricsEngine, metrics_ctx_helper
from ml_factory.models.autoencoder import ModelResult
from ml_factory.training_scripts.training_script_ae_structural import (
    LOSS_STR,
)
from ml_factory.utils.model_saver import ModelSaver
from ml_factory.utils.parameter_tracker import Tracker


@dataclass
class TrainerArgs:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_map: dict[str, Callable]
    include_loss: dict[str, float]
    metric_map: dict[str, Callable] | None = None
    include_metrics: list | None = None
    metrics_on_training: bool = False
    use_amp: bool = False
    ratio_sampler: bool = False
    best_metric: str = LOSS_STR
    best_condition: Literal["le", "ge"] = "le"
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    device: Literal["cpu", "cuda"] = "cuda"


class Trainer:
    def __init__(self, trainer_args: TrainerArgs):
        self.config = trainer_args

        self.__initialize_trainer()

    def __initialize_trainer(self):
        self.device = self.config.device
        self.model = self.config.model.to(self.device)
        self.optimizer = self.config.optimizer
        self.scheduler = self.config.scheduler
        self.scaler = torch.amp.grad_scaler.GradScaler(enabled=self.config.use_amp)
        self.loss_engine = LossEngine(self.config.loss_map, self.config.include_loss)
        self.metric_engine = MetricsEngine(
            metrics_map=self.config.metric_map or {},
            include_metrics=self.config.include_metrics or [],
        )
        self.use_amp = self.config.use_amp
        self.best_model_settings = ModelSaver()
        self.best_score = (
            float("inf") if self.config.best_condition == "le" else float("-inf")
        )

        include_metrics = self.config.include_metrics or []

        self.metric_on_training = (
            self.config.metrics_on_training and len(include_metrics) > 0
        )
        self.has_metrics = len(include_metrics) > 0
        losses = list(self.config.include_loss.keys()) + [LOSS_STR]
        track_eval = losses + include_metrics
        track_train = losses + (include_metrics if self.metric_on_training else [])

        self.include_tracking_eval = track_eval

        self.train_track = Tracker(*track_train, _counter_type=torch.Tensor)
        self.val_track = Tracker(*track_eval, _counter_type=torch.Tensor)

    def _run_epoch(
        self,
        loader: DataLoader,
        track: Tracker,
        training: bool,
        ratio_sampler: bool = False,
        epoch: int = 0,
    ):
        if training:
            self.model.train()
        else:
            self.model.eval()

        track.counter_reset()

        all_y_true = []
        all_y_score = []

        context_manager = torch.enable_grad() if training else torch.no_grad()

        with context_manager:
            for X, y in loader:
                if ratio_sampler and self.config.ratio_sampler:
                    if hasattr(loader.sampler, "set_epoch"):
                        loader.sampler.set_epoch(epoch)

                y = y.to(self.device)
                X = X.to(self.device)

                with torch.autocast(device_type=self.device, enabled=self.use_amp):
                    result: ModelResult = self.model(X)

                    ctx = create_ctx_context(X, y, result)

                    total_loss, components = self.loss_engine.compute(ctx)

                if training:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if self.has_metrics or (training and self.metric_on_training):
                    scores = ((result.recon - X) ** 2).mean(dim=1)
                    all_y_score.append(scores.detach().cpu())
                    all_y_true.append(y.detach().cpu())

                track.counter_update(LOSS_STR, total_loss.detach())
                track.counter_updates(components)

            track.update_logs(_by=len(loader), ignore=self.config.include_metrics or [])

            if (not training and self.has_metrics) or (
                training and self.metric_on_training
            ):
                all_y_score = torch.cat(all_y_score).numpy()
                all_y_true = torch.cat(all_y_true).numpy()
                ctx = metrics_ctx_helper(y_actual=all_y_true, y_score=all_y_score)
                metrics_comp = self.metric_engine.compute(ctx)
                track._logs_update_map(metrics_comp, _by=1)

    def _print_epoch(self, epoch: int, epochs: int):
        print(
            f"EPOCH [{epoch:<3}/{epochs}] | train loss: {self.train_track.latest_log(LOSS_STR):<5.5f}"
            f" | val loss: {self.val_track.latest_log(LOSS_STR):<5.5f} | metric: {self.config.best_metric!r} {self.val_track.latest_log(self.config.best_metric):<5.5f}"
        )

    def __condition_check(self, best_prev, now, condition="le"):
        if condition == "le":
            return best_prev <= now
        else:
            return best_prev >= now

    def _check_best_model(self, epoch):
        if self.__condition_check(
            self.best_score,
            self.val_track.latest_log(self.config.best_metric),
            condition=self.config.best_condition,
        ):
            self.best_model_settings.save(
                model=self.model, optimizer=self.optimizer, epoch=epoch
            )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True,
    ):
        for epoch in range(epochs):
            self._run_epoch(
                train_loader,
                self.train_track,
                training=True,
                ratio_sampler=True,
                epoch=epoch,
            )

            self._run_epoch(
                val_loader, self.val_track, training=False, ratio_sampler=False
            )

            if self.scheduler is not None:
                self.scheduler.step()

            self._check_best_model(epoch)

            if verbose:
                self._print_epoch(epoch, epochs)

        return {
            "train_log": self.train_track,
            "val_log": self.val_track,
            "model": self.best_model_settings,
        }

    def evaluate(self, loader) -> Tracker:
        evaluate_tracker = Tracker(
            *self.include_tracking_eval, _counter_type=torch.Tensor
        )

        self._run_epoch(
            loader=loader, track=evaluate_tracker, training=False, ratio_sampler=False
        )

        return evaluate_tracker
