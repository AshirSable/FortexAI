from dataclasses import dataclass 
from typing import Optional, Literal, List

@dataclass
class Args:
    ae_input_dim: int = 512
    n_hidden_layers: int = 8
    n_experts: int = 8
    expert_k: int = 2
    ae_bottleneck: int = 64
    experts_dim: int = 128
    device: Literal['cuda', 'cpu'] = 'cuda'
