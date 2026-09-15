import torch
from torch import nn


class ModelSaver:
    def __init__(self):
        self.maintain_dict = {"model": {}, "optimizer": None, "epoch": None}

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int | None = None,
        **extra,
    ):
        self.maintain_dict["model"] = {
            k: v.clone() for k, v in model.state_dict().items()
        }
        if optimizer is not None:
            self.maintain_dict["optimizer"] = optimizer.state_dict()

        if epoch is not None:
            self.maintain_dict["epoch"] = epoch

        for ex in extra:
            self.maintain_dict[ex] = extra[ex]

    def get_dict(self):
        return self.maintain_dict
