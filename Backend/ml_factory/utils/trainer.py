import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_factory import MODEL_DIRECTORY_DEV, TRAINING_LOGS_DIR
from ml_factory.evaluation import TOTAL_LOSS
from ml_factory.evaluation.loss import LossEngine, create_ctx_context
from ml_factory.evaluation.metrics import MetricsEngine, metrics_ctx_helper
from ml_factory.models.autoencoder import ModelResult
from ml_factory.training_scripts.scripts_structural import CONFIG
from ml_factory.training_scripts.training_script_ae_structural import (
    LOSS_STR,
)
from ml_factory.utils.early_stopping import EarlyStopping, EarlyStoppingEnum
from ml_factory.utils.model_saver import ModelSaver
from ml_factory.utils.parameter_tracker import Tracker
from ml_factory.utils.plotter import plotting_logs

LR_RATE = "learning_rate"


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
    greater_is_better: bool = False
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    gradient_clipping: float | None = None
    track_lr_rate: bool = False
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_tolerance: float = 1e-6
    early_stopping_metric: str | None = None
    early_stopping_greater_is_better: bool = False
    normalize_bottleneck: bool = False
    weight_mode: Literal["static", "uncertainty", "running_norm"] = "static"
    device: Literal["cpu", "cuda"] = "cuda"

    def _serialize_value(self, val):
        if isinstance(val, partial):
            fn_name = getattr(val.func, "__name__", str(val.func))
            return f"partial({fn_name}, {val.keywords})"
        elif callable(val):
            return getattr(val, "__name__", str(val))
        elif isinstance(val, dict):
            return {k: self._serialize_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [self._serialize_value(v) for v in val]
        elif isinstance(
            val,
            (nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler),
        ):
            return getattr(val, "__name__", str(val))
        return val

    def get_dict(self) -> dict:
        raw_dict = asdict(self)
        return self._serialize_value(raw_dict)


@dataclass
class TrainerOutput:
    train_tracker: Tracker
    val_tracker: Tracker
    model: ModelSaver


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
        self.loss_engine = LossEngine(
            self.config.loss_map,
            self.config.include_loss,
            weight_mode=self.config.weight_mode,
        )
        self.metric_engine = MetricsEngine(
            metrics_map=self.config.metric_map or {},
            include_metrics=self.config.include_metrics or [],
        )
        self.use_amp = self.config.use_amp
        self.best_model_settings = ModelSaver()
        self.best_score = (
            float("-inf") if self.config.greater_is_better else float("inf")
        )

        include_metrics = self.config.include_metrics or []

        self.metric_on_training = (
            self.config.metrics_on_training and len(include_metrics) > 0
        )
        self.has_metrics = len(include_metrics) > 0

        self.do_grad_clip = self.config.gradient_clipping is not None
        self.gradient_clip = self.config.gradient_clipping or 0.0
        losses = list(self.config.include_loss.keys()) + [TOTAL_LOSS]
        self.not_static_weights = self.config.weight_mode != "static"

        include_weights = self._get_weight_names(
            [loss for loss, val in self.config.include_loss.items() if val != 0]
            if self.not_static_weights
            else []
        )
        track_eval = losses + include_metrics
        track_train = (
            losses
            + (include_metrics if self.metric_on_training else [])
            + ([LR_RATE] if self.config.track_lr_rate else [])
            + include_weights
        )

        self.include_tracking_eval = track_eval

        self.train_track = Tracker(*track_train, _counter_type=torch.Tensor)
        self.val_track = Tracker(*track_eval, _counter_type=torch.Tensor)

        self.early_stopper = None
        if self.config.early_stopping:
            self.early_stopper = EarlyStopping(
                patience=self.config.early_stopping_patience,
                tolerance=self.config.early_stopping_tolerance,
                _metric=self.config.early_stopping_metric,
                _greater_is_better=self.config.early_stopping_greater_is_better,
            )

        self._ignore_batch_metric_update = (
            include_metrics
            + ([LR_RATE] if self.config.track_lr_rate else [])
            + include_weights
        )

    def _weight_loss_name(self, loss: str):
        return f"weight_{loss}"

    def _transform_loss_weight_name(self, losses: dict[str, float]):
        return {self._weight_loss_name(loss): val for loss, val in losses.items()}

    def _get_weight_names(self, losses: list[str]):
        return [self._weight_loss_name(loss) for loss in losses]

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
                    if self.config.normalize_bottleneck:
                        result.bottleneck = torch.nn.functional.normalize(
                            result.bottleneck, p=2, dim=1
                        )

                    ctx = create_ctx_context(X, y, result)

                    total_loss, components = self.loss_engine.compute(ctx)

                if training:
                    self.scaler.scale(total_loss).backward()
                    if self.do_grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.gradient_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if self.has_metrics or (training and self.metric_on_training):
                    scores = ((result.recon - X) ** 2).mean(dim=1)
                    all_y_score.append(scores.detach().cpu())
                    all_y_true.append(y.detach().cpu())

                track.counter_update(TOTAL_LOSS, total_loss.detach())
                track.counter_updates(components)

            if self.not_static_weights and training:
                track._logs_update_map(
                    self._transform_loss_weight_name(
                        self.loss_engine.get_current_weights()
                    )
                )
            track.update_logs(_by=len(loader), ignore=self._ignore_batch_metric_update)

            if (not training and self.has_metrics) or (
                training and self.metric_on_training
            ):
                all_y_score = torch.cat(all_y_score).numpy()
                all_y_true = torch.cat(all_y_true).numpy()
                ctx = metrics_ctx_helper(y_actual=all_y_true, y_score=all_y_score)
                metrics_comp = self.metric_engine.compute(ctx)
                track._logs_update_map(metrics_comp, _by=1)

    def _print_epoch(self, epoch: int, epochs: int):
        main_text = (
            f"EPOCH [{epoch:<3}/{epochs}] | train loss: {self.train_track.latest_log(TOTAL_LOSS):<5.5f}"
            f" | val loss: {self.val_track.latest_log(TOTAL_LOSS):<5.5f} | metric: {self.config.best_metric!r} {self.val_track.latest_log(self.config.best_metric):<5.5f}"
        )

        if self.config.track_lr_rate:
            main_text = (
                main_text + f" | learning_rate: {self.scheduler.get_last_lr()[0]}"
            )
        print(main_text)

    def __condition_check(self, now: float):
        if self.config.greater_is_better:
            return self.best_score <= now
        else:
            return self.best_score >= now

    def _check_best_model(self, epoch):
        if self.__condition_check(
            self.val_track.latest_log(self.config.best_metric),
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
                self.train_track._logs_update_single(
                    LR_RATE, value=self.scheduler.get_last_lr()[0], _by=1
                )
                self.scheduler.step()

            self._check_best_model(epoch)

            if self.early_stopper is not None:
                value = self.val_track.latest_log(self.early_stopper.metric)
                stop_signal = self.early_stopper.check(value)

                if stop_signal is EarlyStoppingEnum.stop:
                    print(
                        f"Early Stopping: at epoch {epoch}, by metric: {self.early_stopper.metric!r} = {value} | best value {self.early_stopper._best_metric}"
                    )
                    break

            if verbose:
                self._print_epoch(epoch, epochs)

        return TrainerOutput(
            train_tracker=self.train_track,
            val_tracker=self.val_track,
            model=self.best_model_settings,
        )

    def evaluate(self, loader) -> Tracker:
        evaluate_tracker = Tracker(
            *self.include_tracking_eval, _counter_type=torch.Tensor
        )

        self._run_epoch(
            loader=loader, track=evaluate_tracker, training=False, ratio_sampler=False
        )

        return evaluate_tracker


class TrainingLoop:
    def __init__(
        self,
        training_args: TrainerArgs,
        config: CONFIG,
        _model_save_directory: Path | None = None,
        _logs_save_directory: Path | None = None,
    ):
        self.training_args = training_args
        self.config = config

        self._model_directory = _model_save_directory or MODEL_DIRECTORY_DEV

        self._logs_directory = _logs_save_directory or TRAINING_LOGS_DIR
        self._instance_id = time.time()

    def start(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        verbose: bool = True,
    ) -> None:

        start_time = time.time()

        if verbose:
            self._print_training(start_time)

        trainer = Trainer(trainer_args=self.training_args)

        result = trainer.fit(
            train_loader=train_loader,
            val_loader=validation_loader,
            epochs=self.config.epoch,
            verbose=verbose,
        )
        train_end_time = time.time() - start_time
        eval_tracker = trainer.evaluate(loader=test_loader)

        program_end_time = time.time() - start_time

        if verbose:
            self._print_model_execution_finished(
                start_time, train_end_time, program_end_time
            )

            print("Saving Logs")

        self._model_saving(result)

        if verbose:
            print(
                f"MODEL INSTANCE: {self._instance_id} has been saved to path={self.model_file_saved_path}"
            )

        self.logs_directory_path.mkdir(exist_ok=True)

        self.save_logs(result=result, testing_logs=eval_tracker)

        if verbose:
            print(
                f"MODEL INSTANCE: {self._instance_id} Saved Logs as csv on folder = {self.logs_directory_path}"
            )

        plotting_logs(
            train_logs=result.train_tracker.logs,
            val_logs=result.val_tracker.logs,
            path=self.logs_directory_path / "plot_metrics_loss.png",
        )

        if verbose:
            print(f"MODEL INSTANCE: {self._instance_id} Saved Plot")

        self.write_parameters()
        self.write_training_parameters()

    def write_parameters(self):
        with open(self.logs_directory_path / "parameters.json", "w") as f:
            json.dump(self.config.get_dict(), f, indent=2)

    def write_training_parameters(self):
        with open(self.logs_directory_path / "training_parameters.json", "w") as f:
            json.dump(self.training_args.get_dict(), f, indent=2)

    @property
    def model_file_saved_path(self) -> Path:
        file_path = (
            self._model_directory
            / f"{self.training_args.model.__class__.__name__}_instance_{self._instance_id}"
        )
        return file_path

    @property
    def logs_directory_path(self) -> Path:
        return (
            self._logs_directory
            / f"LOGS_{self.training_args.model.__class__.__name__}_instance_{self._instance_id}"
        )

    def _model_saving(self, result: TrainerOutput):
        torch.save(
            result.model.get_dict(),
            self.model_file_saved_path,
        )

    def save_logs(self, result: TrainerOutput, testing_logs: Tracker):
        result.train_tracker.to_csv(self.logs_directory_path / "train_log.csv")
        result.val_tracker.to_csv(self.logs_directory_path / "validation_log.csv")
        testing_logs.to_csv(self.logs_directory_path / "test_log.csv")

    def _print_training(self, start_time):
        print(
            "#" * 20,
            "\n",
            datetime.now(),
            f"Training Started for MODEL: {self.training_args.model.__class__.__name__}, Instance {start_time}",
        )

    def _print_model_execution_finished(
        self, start_time, train_end_time, program_end_time
    ):
        print(
            f"MODEL TRAINING: Instance {start_time} {self.training_args.model.__class__.__name__} Training Finished in {train_end_time} seconds, Program Finished in {program_end_time}"
        )
