from copy import copy
from os import wait

import torch

from ml_factory.evaluation import (
    AUPRC_METRIC,
    CONTRASTIVE_LOSS,
    DIVERSITY_LOSS,
    MSE_LOSS,
    OE_LOSS,
    ROCAUC_METRIC,
    ROUTER_LOAD_BALANCE_LOSS,
    TOTAL_LOSS,
)
from ml_factory.evaluation.loss import LOSS_MAP
from ml_factory.evaluation.metrics import METRICS_MAP
from ml_factory.evaluation.utils import EvaluationMapMapper, IncludeEvaluationMapper
from ml_factory.models import Args
from ml_factory.models.autoencoder import BaseNormalAutoEncoder, NormalityAE
from ml_factory.training_scripts.scripts_structural import (
    get_full_data,
)
from ml_factory.utils import CONFIG
from ml_factory.utils.trainer import TrainerArgs, TrainingLoop

SEED = 3123
BATCH_SIZE = 512
LEARNING_RATE = 0.001


def attack_trainings():
    loss_map = EvaluationMapMapper(LOSS_MAP)
    loss_map.add_kwargs(MSE_LOSS, reduction="mean").add_kwargs(
        OE_LOSS, return_components=False, oe_margin=2.0, oe_weight=1.0
    ).add_kwargs(CONTRASTIVE_LOSS, margin=1.0)
    include_loss = IncludeEvaluationMapper([MSE_LOSS, OE_LOSS]).change_score(
        OE_LOSS, 2.0
    )

    include_loss_normality_contrastive = IncludeEvaluationMapper(
        [MSE_LOSS, CONTRASTIVE_LOSS]
    ).change_score(CONTRASTIVE_LOSS, 1.5)

    include_loss_normality_oe_diversity = IncludeEvaluationMapper(
        [MSE_LOSS, OE_LOSS, DIVERSITY_LOSS]
    ).change_score(OE_LOSS, 2.0)

    include_loss_normality_oe_diversity_router = IncludeEvaluationMapper(
        [MSE_LOSS, OE_LOSS, DIVERSITY_LOSS, ROUTER_LOAD_BALANCE_LOSS]
    ).change_score(OE_LOSS, 2.0)
    include_loss_normality_contrastive_diversity = IncludeEvaluationMapper(
        [MSE_LOSS, CONTRASTIVE_LOSS, DIVERSITY_LOSS]
    ).change_score(CONTRASTIVE_LOSS, 1.5)
    include_loss_normality_oe_contrastive = (
        IncludeEvaluationMapper([MSE_LOSS, OE_LOSS, CONTRASTIVE_LOSS])
        .change_score(OE_LOSS, 2.0)
        .change_score(CONTRASTIVE_LOSS, 1.5)
    )

    include_loss_normality_oe_contrastive_diversity = (
        IncludeEvaluationMapper([MSE_LOSS, OE_LOSS, CONTRASTIVE_LOSS, DIVERSITY_LOSS])
        .change_score(OE_LOSS, 2.0)
        .change_score(CONTRASTIVE_LOSS, 1.5)
    )

    include_loss_normality_oe_contrastive_router = (
        IncludeEvaluationMapper(
            [MSE_LOSS, OE_LOSS, CONTRASTIVE_LOSS, ROUTER_LOAD_BALANCE_LOSS]
        )
        .change_score(OE_LOSS, 2.0)
        .change_score(CONTRASTIVE_LOSS, 1.5)
    )

    include_loss_normality_contrastive_router = IncludeEvaluationMapper(
        [MSE_LOSS, CONTRASTIVE_LOSS, ROUTER_LOAD_BALANCE_LOSS]
    ).change_score(CONTRASTIVE_LOSS, 1.5)

    include_loss_normality_oe_router = IncludeEvaluationMapper(
        [MSE_LOSS, OE_LOSS, ROUTER_LOAD_BALANCE_LOSS]
    ).change_score(OE_LOSS, 2.0)

    include_loss_normality_oe_contrastive_diversity_router = (
        IncludeEvaluationMapper(
            [
                MSE_LOSS,
                OE_LOSS,
                ROUTER_LOAD_BALANCE_LOSS,
                DIVERSITY_LOSS,
                CONTRASTIVE_LOSS,
            ]
        )
        .change_score(OE_LOSS, 2.0)
        .change_score(CONTRASTIVE_LOSS, 1.5)
    )

    metrics = METRICS_MAP

    include_metrics = [ROCAUC_METRIC, AUPRC_METRIC]

    base_config = CONFIG(
        loss_map=loss_map.get_map(),
        include_loss=include_loss.get_metrics_map(),
        metrics_map=metrics,
        include_metrics=include_metrics,
        best_metric=AUPRC_METRIC,
        greater_is_better=True,
        batch_size=BATCH_SIZE,
        seed=SEED,
        ratio_sampler=True,
        attack_ratio=0.2,
        ae_embedding_size=100,
    )

    base_model_losses = [
        # include_loss,
        # include_loss_normality_contrastive,
        include_loss_normality_oe_contrastive,
    ]

    all_include_losses = [
        # include_loss,
        # include_loss_normality_contrastive,
        # include_loss_normality_oe_contrastive,
        # include_loss_normality_oe_diversity,
        # include_loss_normality_contrastive_diversity,
        # include_loss_normality_oe_router,
        # include_loss_normality_contrastive_router,
        # include_loss_normality_oe_contrastive_router,
        # include_loss_normality_oe_contrastive_diversity,
        include_loss_normality_oe_contrastive_diversity_router,
        # include_loss_normality_oe_diversity_router,
    ]

    dataloaders = get_full_data(base_config)
    args = Args(ae_bottleneck=base_config.ae_embedding_size, expert_k=4)

    for il in base_model_losses:
        custom_config = copy(base_config)
        custom_config.include_loss = il.get_metrics_map()
        model = BaseNormalAutoEncoder(
            dataloaders.x_dim[-1], args, with_dropout=True
        ).to(custom_config.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=custom_config.epoch
        )

        train_args = TrainerArgs(
            model=model,
            optimizer=optimizer,
            loss_map=custom_config.loss_map,
            include_loss=custom_config.include_loss,
            metric_map=custom_config.metrics_map,
            include_metrics=custom_config.include_metrics,
            use_amp=True,
            best_metric=custom_config.best_metric,
            greater_is_better=custom_config.greater_is_better,
            scheduler=scheduler,
            device=custom_config.device,
            track_lr_rate=True,
            early_stopping=True,
            early_stopping_metric=TOTAL_LOSS,
            early_stopping_greater_is_better=False,
            early_stopping_patience=20,
            early_stopping_tolerance=1e-6,
            normalize_bottleneck=True,
            weight_mode="running_norm",
        )

        loop = TrainingLoop(training_args=train_args, config=custom_config)

        loop.start(
            train_loader=dataloaders.train,
            validation_loader=dataloaders.validation,
            test_loader=dataloaders.test,
            verbose=True,
        )

    for il in all_include_losses:
        custom_config = copy(base_config)
        custom_config.include_loss = il.get_metrics_map()

        normal_model = NormalityAE(
            dataloaders.x_dim[-1], args, with_dropout=True, _mode_2=True
        ).to(custom_config.device)
        optimizer = torch.optim.AdamW(normal_model.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=custom_config.epoch
        )

        train_args = TrainerArgs(
            model=normal_model,
            optimizer=optimizer,
            loss_map=custom_config.loss_map,
            include_loss=custom_config.include_loss,
            metric_map=custom_config.metrics_map,
            include_metrics=custom_config.include_metrics,
            use_amp=True,
            best_metric=custom_config.best_metric,
            greater_is_better=custom_config.greater_is_better,
            scheduler=scheduler,
            device=custom_config.device,
            track_lr_rate=True,
            early_stopping=True,
            early_stopping_metric=TOTAL_LOSS,
            early_stopping_greater_is_better=False,
            early_stopping_patience=20,
            early_stopping_tolerance=1e-6,
            normalize_bottleneck=True,
            weight_mode="running_norm",
        )

        loop = TrainingLoop(training_args=train_args, config=custom_config)

        loop.start(
            train_loader=dataloaders.train,
            validation_loader=dataloaders.validation,
            test_loader=dataloaders.test,
            verbose=True,
        )


def attack_trainings_alpha():
    loss_map = EvaluationMapMapper(LOSS_MAP)
    loss_map.add_kwargs(MSE_LOSS, reduction="mean").add_kwargs(
        OE_LOSS, return_components=False, oe_margin=2.0, oe_weight=1.0
    ).add_kwargs(CONTRASTIVE_LOSS, margin=1.0)
    include_loss = IncludeEvaluationMapper([MSE_LOSS, OE_LOSS])

    include_loss_normality_contrastive = IncludeEvaluationMapper(
        [MSE_LOSS, CONTRASTIVE_LOSS]
    )

    include_loss_normality_oe_diversity = IncludeEvaluationMapper(
        [MSE_LOSS, OE_LOSS, DIVERSITY_LOSS]
    )

    include_loss_normality_oe_diversity_router = IncludeEvaluationMapper(
        [MSE_LOSS, OE_LOSS, DIVERSITY_LOSS, ROUTER_LOAD_BALANCE_LOSS]
    )
    include_loss_normality_contrastive_diversity = IncludeEvaluationMapper(
        [MSE_LOSS, CONTRASTIVE_LOSS, DIVERSITY_LOSS]
    )
    include_loss_normality_oe_contrastive = IncludeEvaluationMapper(
        [MSE_LOSS, OE_LOSS, CONTRASTIVE_LOSS]
    )

    include_loss_normality_oe_contrastive_diversity = IncludeEvaluationMapper(
        [MSE_LOSS, OE_LOSS, CONTRASTIVE_LOSS, DIVERSITY_LOSS]
    )

    include_loss_normality_oe_contrastive_router = IncludeEvaluationMapper(
        [MSE_LOSS, OE_LOSS, CONTRASTIVE_LOSS, ROUTER_LOAD_BALANCE_LOSS]
    )

    include_loss_normality_contrastive_router = IncludeEvaluationMapper(
        [MSE_LOSS, CONTRASTIVE_LOSS, ROUTER_LOAD_BALANCE_LOSS]
    )
    include_loss_normality_oe_router = IncludeEvaluationMapper(
        [MSE_LOSS, OE_LOSS, ROUTER_LOAD_BALANCE_LOSS]
    )
    include_loss_normality_oe_contrastive_diversity_router = IncludeEvaluationMapper(
        [
            MSE_LOSS,
            OE_LOSS,
            ROUTER_LOAD_BALANCE_LOSS,
            DIVERSITY_LOSS,
            CONTRASTIVE_LOSS,
        ]
    )

    metrics = METRICS_MAP

    include_metrics = [ROCAUC_METRIC, AUPRC_METRIC]

    base_config = CONFIG(
        loss_map=loss_map.get_map(),
        include_loss=include_loss.get_metrics_map(),
        metrics_map=metrics,
        include_metrics=include_metrics,
        best_metric=AUPRC_METRIC,
        greater_is_better=True,
        batch_size=BATCH_SIZE,
        seed=SEED,
        ratio_sampler=True,
        attack_ratio=0.2,
        ae_embedding_size=100,
    )

    base_model_losses = [
        include_loss,
        # include_loss_normality_contrastive,
        # include_loss_normality_oe_contrastive,
    ]

    all_include_losses = [
        # include_loss,
        # include_loss_normality_contrastive,
        # include_loss_normality_oe_contrastive,
        # include_loss_normality_oe_diversity,
        # include_loss_normality_contrastive_diversity,
        # include_loss_normality_oe_router,
        # include_loss_normality_contrastive_router,
        # include_loss_normality_oe_contrastive_router,
        # include_loss_normality_oe_contrastive_diversity,
        # include_loss_normality_oe_contrastive_diversity_router,
        include_loss_normality_oe_diversity_router,
    ]

    dataloaders = get_full_data(base_config)
    args = Args(ae_bottleneck=base_config.ae_embedding_size, expert_k=4)

    for il in base_model_losses:
        custom_config = copy(base_config)
        custom_config.include_loss = il.get_metrics_map()
        model = BaseNormalAutoEncoder(
            dataloaders.x_dim[-1], args, with_dropout=True
        ).to(custom_config.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=custom_config.epoch
        )

        train_args = TrainerArgs(
            model=model,
            optimizer=optimizer,
            loss_map=custom_config.loss_map,
            include_loss=custom_config.include_loss,
            metric_map=custom_config.metrics_map,
            include_metrics=custom_config.include_metrics,
            use_amp=True,
            best_metric=custom_config.best_metric,
            greater_is_better=custom_config.greater_is_better,
            scheduler=scheduler,
            device=custom_config.device,
            track_lr_rate=True,
            early_stopping=True,
            early_stopping_metric=AUPRC_METRIC,
            early_stopping_greater_is_better=True,
            early_stopping_patience=20,
            early_stopping_tolerance=1e-6,
            normalize_bottleneck=True,
            weight_mode="running_norm",
            gradient_clipping=5.0,
        )

        loop = TrainingLoop(training_args=train_args, config=custom_config)

        loop.start(
            train_loader=dataloaders.train,
            validation_loader=dataloaders.validation,
            test_loader=dataloaders.test,
            verbose=True,
        )

    for il in all_include_losses:
        custom_config = copy(base_config)
        custom_config.include_loss = il.get_metrics_map()

        normal_model = NormalityAE(
            dataloaders.x_dim[-1], args, with_dropout=True, _mode_2=True
        ).to(custom_config.device)
        optimizer = torch.optim.AdamW(normal_model.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=custom_config.epoch
        )

        train_args = TrainerArgs(
            model=normal_model,
            optimizer=optimizer,
            loss_map=custom_config.loss_map,
            include_loss=custom_config.include_loss,
            metric_map=custom_config.metrics_map,
            include_metrics=custom_config.include_metrics,
            use_amp=True,
            best_metric=custom_config.best_metric,
            greater_is_better=custom_config.greater_is_better,
            scheduler=scheduler,
            device=custom_config.device,
            track_lr_rate=True,
            early_stopping=True,
            early_stopping_metric=AUPRC_METRIC,
            early_stopping_greater_is_better=True,
            early_stopping_patience=20,
            early_stopping_tolerance=1e-6,
            normalize_bottleneck=True,
            weight_mode="running_norm",
            gradient_clipping=5.0,
        )

        loop = TrainingLoop(training_args=train_args, config=custom_config)

        loop.start(
            train_loader=dataloaders.train,
            validation_loader=dataloaders.validation,
            test_loader=dataloaders.test,
            verbose=True,
        )


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    attack_trainings_alpha()
