from dataclasses import dataclass
from json import decoder
from typing import Optional

import torch
from torch import nn

from ml_factory.models import Args


@dataclass
class ModelResult:
    recon: torch.Tensor
    experts: torch.Tensor
    bottleneck: torch.Tensor
    router: torch.Tensor
    k_router: Optional[int] = None


class BaseNormalAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, args: Args, with_dropout=False):
        super().__init__()
        if args.ae_bottleneck > 100:
            raise AssertionError(
                f"bottleneck cannot be greater than 100, currently got: {args.ae_bottleneck}"
            )

        dropout_value = args.dropout if with_dropout else 0.0

        self.__encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(dropout_value),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(dropout_value),
            nn.Linear(256, 100),
            nn.SiLU(),
            nn.Dropout(dropout_value),
            nn.Linear(100, args.ae_bottleneck),
        )

        self.__decoder = nn.Sequential(
            nn.Linear(args.ae_bottleneck, 100),
            nn.SiLU(),
            nn.Dropout(dropout_value),
            nn.Linear(100, 256),
            nn.SiLU(),
            nn.Dropout(dropout_value),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Dropout(dropout_value),
            nn.Linear(512, input_dim),
        )

    def forward(self, X: torch.Tensor):
        X = self.encode(X)
        return ModelResult(
            recon=self.decode(X),
            bottleneck=X,
            experts=torch.tensor([]),
            router=torch.tensor([]),
            k_router=None,
        )

    def __call__(self, *args, **kwargs) -> ModelResult:
        return super().__call__(*args, **kwargs)

    def encode(self, X: torch.Tensor):
        return self.__encoder(X)

    def decode(self, encoded_X: torch.Tensor):
        return self.__decoder(encoded_X)


class MLP(nn.Module):
    def __init__(self, input_dim: int, args: Args, with_dropout: bool = False):
        super().__init__()
        dropout_value = args.dropout if with_dropout else 0.0
        if input_dim < args.ae_bottleneck:
            raise ValueError(
                f"input_dim must not be less than the bottleneck: "
                f"input_dim={input_dim}, bottleneck={args.ae_bottleneck}"
            )
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.RMSNorm(input_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_value),
            nn.Linear(input_dim // 2, args.ae_bottleneck),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class NormalityAE(nn.Module):
    def __init__(
        self, input_dim: int, args: Args, with_dropout=False, _mode_2: bool = False
    ):
        super().__init__()
        dropout_value = args.dropout if with_dropout else 0.0
        if _mode_2:
            dims = [input_dim, 512, 256, 128, args.ae_bottleneck]

        else:
            dims = [input_dim, 512, 256, args.ae_bottleneck]
        self.k_expert = args.expert_k
        self.n_experts = args.n_experts

        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if (i + 1) != (len(dims) - 1):
                encoder_layers.append(nn.SiLU())

            encoder_layers.append(nn.Dropout(dropout_value))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in reversed(range(len(dims) - 1)):
            decoder_layers.append(nn.Linear(dims[i + 1], dims[i]))
            decoder_layers.append(nn.Dropout(dropout_value))
            if i != 0:
                decoder_layers.append(nn.SiLU())
        self.decoder = nn.Sequential(*decoder_layers)

        self.router = nn.Linear(dims[-1], self.n_experts)
        self.experts = nn.ModuleList(
            [MLP(dims[-1], args, with_dropout) for _ in range(self.n_experts)]
        )

    def forward(self, X: torch.Tensor):
        x = self.encoder(X)

        router_logits = self.router(x)
        top_vals, top_idx = router_logits.topk(self.k_expert, dim=-1)
        weights = torch.softmax(top_vals, dim=-1)

        all_expert_out = torch.stack([e(x) for e in self.experts], dim=1)
        gathered = torch.gather(
            all_expert_out,
            1,
            top_idx.unsqueeze(-1).expand(-1, -1, all_expert_out.size(-1)),
        )
        combined = (gathered * weights.unsqueeze(-1)).sum(dim=1)

        recon = self.decoder(combined)
        return ModelResult(
            recon=recon,
            experts=all_expert_out,
            bottleneck=combined,
            router=router_logits,
            k_router=self.k_expert,
        )

    def __call__(self, *args, **kwargs) -> ModelResult:
        return super().__call__(*args, **kwargs)

    def encode(self, X: torch.Tensor):
        return self.encoder(X)
