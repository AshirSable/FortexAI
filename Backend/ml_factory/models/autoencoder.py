import torch
import torch.nn as nn
from ml_factory.models import Args


class MLP(nn.Module):
    def __init__(self, input_dim: int, args: Args):
        super().__init__()
        if input_dim < args.ae_bottleneck:
            raise ValueError(
                f'input_dim must not be less than the bottleneck: '
                f'input_dim={input_dim}, bottleneck={args.ae_bottleneck}'
            )
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.RMSNorm(input_dim // 2),
            nn.SiLU(),
            nn.Linear(input_dim // 2, args.ae_bottleneck),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)


class NormalityAE(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        dims = [args.ae_input_dim, 512, 256, 128]
        self.k_expert = args.expert_k
        self.n_experts = args.n_experts

        encoder_layers = []
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            encoder_layers.append(nn.SiLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in reversed(range(len(dims) - 1)):
            decoder_layers.append(nn.Linear(dims[i + 1], dims[i]))
            if i != 0:
                decoder_layers.append(nn.SiLU())
        self.decoder = nn.Sequential(*decoder_layers)

        self.router = nn.Linear(dims[-1], self.n_experts)
        self.experts = nn.ModuleList([MLP(dims[-1], args) for _ in range(self.n_experts)])

    def forward(self, X: torch.Tensor):
        x = self.encoder(X)  # (batch, bottleneck_in)

        router_logits = self.router(x)  # (batch, n_experts)
        top_vals, top_idx = router_logits.topk(self.k_expert, dim=-1)  # (batch, k)
        weights = torch.softmax(top_vals, dim=-1)  # (batch, k)

        all_expert_out = torch.stack([e(x) for e in self.experts], dim=1)  # (batch, n_experts, ae_bottleneck)
        gathered = torch.gather(
            all_expert_out, 1,
            top_idx.unsqueeze(-1).expand(-1, -1, all_expert_out.size(-1))
        )  # (batch, k, ae_bottleneck)
        combined = (gathered * weights.unsqueeze(-1)).sum(dim=1)  # (batch, ae_bottleneck)

        recon = self.decoder(combined)
        return recon, all_expert_out  # return all expert outs once, for diversity loss

    def encode(self, X: torch.Tensor):
        return self.encoder(X)
