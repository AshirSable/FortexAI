from copy import copy

import torch

from ml_factory.evaluation import (
    AUPRC_METRIC,
    CONTRASTIVE_LOSS,
    DIVERSITY_LOSS,
    MSE_LOSS,
    OE_LOSS,
    ROCAUC_METRIC,
    ROUTER_LOAD_BALANCE_LOSS,
)
from ml_factory.evaluation.loss import LOSS_MAP
from ml_factory.evaluation.metrics import METRICS_MAP
from ml_factory.evaluation.utils import EvaluationMapMapper, IncludeEvaluationMapper
from ml_factory.models import Args
from ml_factory.models.autoencoder import BaseNormalAutoEncoder, NormalityAE
from ml_factory.training_scripts.scripts_structural import (
    CONFIG,
    autoencoder_training,
    get_benign_data,
    get_full_data,
)

SEED = 3128
BATCH_SIZE = 200
LEARNING_RATE = 0.001


def benign_trainings():

    loss_map = EvaluationMapMapper(LOSS_MAP)
    loss_map.add_kwargs(MSE_LOSS, reduction="mean")
    include_loss = IncludeEvaluationMapper([MSE_LOSS])

    include_loss_normality_diversity = IncludeEvaluationMapper(
        [MSE_LOSS, DIVERSITY_LOSS]
    )

    include_loss_normality_diversity_router = IncludeEvaluationMapper(
        [MSE_LOSS, DIVERSITY_LOSS, ROUTER_LOAD_BALANCE_LOSS]
    )

    benign_base_ae_config = CONFIG(
        loss_map=loss_map.get_map(),
        include_loss=include_loss.get_metrics_map(),
        metrics_map={},
        include_metrics=[],
        best_metric=MSE_LOSS,
        best_metric_condition="le",
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    normality_diversity_config = CONFIG(
        loss_map=loss_map.get_map(),
        include_loss=include_loss_normality_diversity.get_metrics_map(),
        metrics_map={},
        include_metrics=[],
        best_metric=MSE_LOSS,
        best_metric_condition="le",
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    normality_diversity_router_config = CONFIG(
        loss_map=loss_map.get_map(),
        include_loss=include_loss_normality_diversity_router.get_metrics_map(),
        metrics_map={},
        include_metrics=[],
        best_metric=MSE_LOSS,
        best_metric_condition="le",
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    dataloaders = get_benign_data(benign_base_ae_config)
    args = Args()

    model = BaseNormalAutoEncoder(dataloaders.x_dim[-1], args).to(
        benign_base_ae_config.device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    autoencoder_training(dataloaders, model, optimizer, benign_base_ae_config)

    normal_model = NormalityAE(dataloaders.x_dim[-1], args).to(
        benign_base_ae_config.device
    )
    optimizer = torch.optim.AdamW(normal_model.parameters(), lr=0.001)

    autoencoder_training(dataloaders, normal_model, optimizer, benign_base_ae_config)

    normal_model = NormalityAE(dataloaders.x_dim[-1], args).to(
        benign_base_ae_config.device
    )
    optimizer = torch.optim.AdamW(normal_model.parameters(), lr=0.001)
    autoencoder_training(
        dataloaders, normal_model, optimizer, normality_diversity_config
    )

    normal_model = NormalityAE(dataloaders.x_dim[-1], args).to(
        benign_base_ae_config.device
    )
    optimizer = torch.optim.AdamW(normal_model.parameters(), lr=0.001)

    autoencoder_training(
        dataloaders, normal_model, optimizer, normality_diversity_router_config
    )


def attack_trainings():
    loss_map = EvaluationMapMapper(LOSS_MAP)
    loss_map.add_kwargs(MSE_LOSS, reduction="mean").add_kwargs(
        OE_LOSS, return_components=False
    )
    include_loss = IncludeEvaluationMapper([MSE_LOSS, OE_LOSS])

    include_loss_normality_contrastive = IncludeEvaluationMapper(
        [MSE_LOSS, CONTRASTIVE_LOSS]
    )

    include_loss_normality_oe_diversity = IncludeEvaluationMapper(
        [MSE_LOSS, OE_LOSS, DIVERSITY_LOSS]
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
        [MSE_LOSS, OE_LOSS, ROUTER_LOAD_BALANCE_LOSS, DIVERSITY_LOSS, CONTRASTIVE_LOSS]
    )

    metrics = METRICS_MAP

    include_metrics = [ROCAUC_METRIC, AUPRC_METRIC]

    base_config = CONFIG(
        loss_map=loss_map.get_map(),
        include_loss=include_loss.get_metrics_map(),
        metrics_map=metrics,
        include_metrics=include_metrics,
        best_metric=AUPRC_METRIC,
        best_metric_condition="ge",
        batch_size=BATCH_SIZE,
        seed=SEED,
        ratio_sampler=True,
        attack_ratio=0.2,
    )

    base_model_losses = [
        include_loss,
        include_loss_normality_contrastive,
        include_loss_normality_oe_contrastive,
    ]

    all_include_losses = [
        include_loss,
        include_loss_normality_contrastive,
        include_loss_normality_oe_contrastive,
        include_loss_normality_oe_diversity,
        include_loss_normality_contrastive_diversity,
        include_loss_normality_oe_router,
        include_loss_normality_contrastive_router,
        include_loss_normality_oe_contrastive_router,
        include_loss_normality_oe_contrastive_diversity,
        include_loss_normality_oe_contrastive_diversity_router,
    ]

    dataloaders = get_full_data(base_config)
    args = Args()

    for il in base_model_losses:
        custom_config = copy(base_config)
        custom_config.include_loss = il.get_metrics_map()
        model = BaseNormalAutoEncoder(dataloaders.x_dim[-1], args).to(
            custom_config.device
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        autoencoder_training(dataloaders, model, optimizer, custom_config)

    for il in all_include_losses:
        custom_config = copy(base_config)
        custom_config.include_loss = il.get_metrics_map()

        normal_model = NormalityAE(dataloaders.x_dim[-1], args).to(custom_config.device)
        optimizer = torch.optim.AdamW(normal_model.parameters(), lr=0.001)

        autoencoder_training(dataloaders, normal_model, optimizer, custom_config)


if __name__ == "__main__":
    # benign_trainings()
    attack_trainings()
