import inspect
from dataclasses import fields, is_dataclass
import re
from typing import Callable, Literal

import torch
import torch.nn.functional as F

from ml_factory.evaluation.abstract import EvaluationEngine
from ml_factory.models.autoencoder import ModelResult


def create_ctx_context(X: torch.Tensor, y: torch.Tensor, result: ModelResult) -> dict:
    if is_dataclass(result):
        ctx = {f.name: getattr(result, f.name) for f in fields(result)}
    else:
        ctx = vars(result).copy()

    ctx["X"] = X
    ctx["y"] = y

    ctx["X_hat"] = ctx.get("recon")
    ctx["z"] = ctx.get("bottleneck")
    ctx["experts"] = ctx.get("experts")
    ctx["k"] = ctx.get("k_router")
    return ctx


def _sanitize_name(name: str) -> str:
    """Buffer/parameter names can't contain '.' or other special chars."""
    return re.sub(r"\W", "_", name)


class LossEngine(torch.nn.Module, EvaluationEngine):
    def __init__(
        self,
        loss_map: dict[str, Callable],
        include_loss: dict[str, float],
        weight_mode: Literal["static", "uncertainty", "running_norm"] = "static",
        norm_momentum: float = 0.9,
    ):
        super().__init__()
        self.loss_map = loss_map
        self.include_loss = include_loss
        self.weight_mode: Literal["static", "uncertainty", "running_norm"] = weight_mode
        self.norm_momentum = norm_momentum

        self.__param_cache = {
            name: set(inspect.signature(fn).parameters.keys())
            for name, fn in loss_map.items()
        }

        active_losses = [name for name, w in include_loss.items() if w > 0]
        self._name_map = {name: _sanitize_name(name) for name in active_losses}
        if weight_mode == "uncertainty":
            self.log_vars = torch.nn.ParameterDict(
                {name: torch.nn.Parameter(torch.zeros(())) for name in active_losses}
            )
        elif weight_mode == "running_norm":
            for name in active_losses:
                self.register_buffer(f"_norm_{self._name_map[name]}", torch.tensor(1.0))

    def _get_norm_buf(self, loss_name: str) -> torch.Tensor:
        return getattr(self, f"_norm_{self._name_map[loss_name]}")

    def _set_norm_buf(self, loss_name: str, value: torch.Tensor) -> None:
        setattr(self, f"_norm_{self._name_map[loss_name]}", value)

    def _weighted(self, loss_name: str, loss_val: torch.Tensor) -> torch.Tensor:
        static_weight = self.include_loss[loss_name]

        if self.weight_mode == "static":
            return loss_val * static_weight

        if self.weight_mode == "uncertainty":
            log_var = self.log_vars[loss_name]
            precision = torch.exp(-log_var)

            return static_weight * (precision * loss_val + log_var)

        if self.weight_mode == "running_norm":
            with torch.no_grad():
                running = self._get_norm_buf(loss_name)
                updated = (
                    self.norm_momentum * running
                    + (1 - self.norm_momentum) * loss_val.detach().float()
                )
                self._set_norm_buf(loss_name, updated)

                normalizer = updated.clamp(min=1e-2)
                effective_denom = (static_weight / normalizer).clamp(max=100)
            return loss_val * effective_denom

        raise ValueError(f"unknown weight_mode: {self.weight_mode}")

    def compute(self, ctx: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total_loss = torch.tensor(0.0, device=ctx["X"].device)
        component_loss = {}

        for loss_name, weight in self.include_loss.items():
            if weight <= 0 or loss_name not in self.loss_map:
                continue

            loss_fn = self.loss_map[loss_name]
            fn_params = self.__param_cache[loss_name]

            filtered_kwargs = {k: v for k, v in ctx.items() if k in fn_params}

            loss_val = loss_fn(**filtered_kwargs)

            if torch.isnan(loss_val).any() or torch.isinf(loss_val).any():
                print(f"{loss_name} bad loss val: {loss_val}")
            weighted = self._weighted(loss_name=loss_name, loss_val=loss_val)
            total_loss = total_loss + weighted
            component_loss[loss_name] = loss_val.detach()

        return total_loss, component_loss

    def get_current_weights(self) -> dict[str, float]:
        weights = {}
        for name, static_weight in self.include_loss.items():
            if static_weight <= 0 or name not in self.loss_map:
                continue

            if self.weight_mode == "static":
                weights[name] = static_weight
            elif self.weight_mode == "uncertainty":
                weights[name] = torch.exp(-self.log_vars[name]).detach().item()
            elif self.weight_mode == "running_norm":
                weights[name] = (
                    (static_weight / self._get_norm_buf(name).clamp(min=1e-2).detach())
                    .clamp(max=100.0)
                    .item()
                )

        return weights


class ClassSplitMetricsEngine:
    def __init__(self, metrics_fns: dict[str, Callable]):
        self.metrics_fns = metrics_fns

    def compute(self, ctx: dict) -> dict[str, torch.Tensor]:
        out = {}
        for fn in self.metrics_fns.values():
            out.update(
                fn(
                    **{
                        k: v
                        for k, v in ctx.items()
                        if k in inspect.signature(fn).parameters
                    }
                )
            )

        return out


def mse_metrics(X_hat, X, y, **_) -> dict[str, torch.Tensor]:
    per_sample = ((X_hat - X) ** 2).mean(dim=1)

    benign_mask, attack_mask = y == 0, y == 1

    return {
        "mse_benign_sum": per_sample[benign_mask].sum(),
        "mse_benign_count": benign_mask.sum().float(),
        "mse_attack_sum": per_sample[attack_mask].sum(),
        "mse_attack_count": attack_mask.sum().float(),
    }


def oe_metric(X_hat, X, y, **_) -> dict[str, torch.Tensor]:
    per_sample = ((X_hat - X) ** 2).mean(dim=1)

    benign_mask, attack_mask = y == 0, y == 1

    return {
        "oe_benign_sum": per_sample[benign_mask].sum(),
        "oe_benign_count": benign_mask.sum().float(),
        "oe_attack_sum": per_sample[attack_mask].sum(),
        "oe_attack_count": attack_mask.sum().float(),
    }


def compute_oe_loss(
    X_hat,
    X,
    y,
    criterion=torch.nn.MSELoss(reduction="none"),
    oe_weight=1.0,
    oe_margin=1.0,
    return_components: bool = True,
    **_,
):
    per_sample = criterion(X_hat, X).mean(dim=1)

    benign_mask = y == 0
    attack_mask = y == 1

    benign_loss = (
        per_sample[benign_mask].mean()
        if benign_mask.any()
        else torch.tensor(0.0, device=X.device)
    )
    if attack_mask.any():
        attack_loss = torch.clamp(oe_margin - per_sample[attack_mask], min=0).mean()

    else:
        attack_loss = torch.tensor(0.0, device=X.device)

    total = benign_loss + oe_weight * attack_loss

    if return_components:
        return total, benign_loss.detach(), attack_loss.detach()
    return total


def diversity_loss(experts: torch.Tensor, **_):
    n_experts = experts.size(1)

    normed = F.normalize(experts, dim=-1)
    sim_matrix = torch.einsum("bnd,bmd->bnm", normed, normed)

    mask = ~torch.eye(n_experts, dtype=torch.bool, device=experts.device)

    off_diag_sim = sim_matrix[:, mask].view(experts.size(0), -1)

    return off_diag_sim.pow(2).mean()


def contrastive_loss(z: torch.Tensor, y: torch.Tensor, margin=1.0, **_):
    benign_mask = y == 0
    attack_mask = y == 1

    if not benign_mask.any():
        return torch.tensor(0.0, device=z.device)

    benign_centroid = z[benign_mask].mean(dim=0, keepdim=True).detach()

    benign_dist = ((z[benign_mask] - benign_centroid) ** 2).sum(dim=1)

    benign_term = benign_dist.mean()

    if attack_mask.any():
        attack_dist = (
            ((z[attack_mask] - benign_centroid) ** 2).sum(dim=1) + 1e-4
        ).sqrt()
        attack_term = torch.clamp(margin - attack_dist, min=0).pow(2).mean()
    else:
        attack_term = torch.tensor(0.0, device=z.device)

    return benign_term + attack_term


def router_load_balance_loss(router: torch.Tensor, k: int, **_) -> torch.Tensor:
    n_experts = router.size(-1)

    router_probs = F.softmax(router, dim=-1)

    mean_prob_per_expert = router_probs.mean(dim=0)

    _, topk_idx = router.topk(k, dim=-1)

    one_hot = torch.zeros_like(router_probs)

    one_hot.scatter_(1, topk_idx, 1.0)

    fraction_selected = one_hot.mean(dim=0)

    loss = n_experts * (fraction_selected * mean_prob_per_expert).sum()

    return loss


def mse_loss(X_hat, X, reduction="none", **_):
    return F.mse_loss(X_hat, X, reduction=reduction)


from ml_factory.evaluation import (
    CONTRASTIVE_LOSS,
    DIVERSITY_LOSS,
    MSE_LOSS,
    OE_LOSS,
    ROUTER_LOAD_BALANCE_LOSS,
)

LOSS_MAP = {
    OE_LOSS: compute_oe_loss,
    DIVERSITY_LOSS: diversity_loss,
    CONTRASTIVE_LOSS: contrastive_loss,
    ROUTER_LOAD_BALANCE_LOSS: router_load_balance_loss,
    MSE_LOSS: mse_loss,
}
