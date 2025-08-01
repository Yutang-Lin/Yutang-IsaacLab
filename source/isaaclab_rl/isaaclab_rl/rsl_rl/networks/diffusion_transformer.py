import torch
import torch.nn as nn
from .dit import DiT

class DiffusionTransformer(nn.Module):
    def __init__(self, action_dim, 
                 condition_dim,
                 timestep_dim,
                 d_model,
                 num_heads,
                 hidden_dim,
                 num_layers,
                 condition_tokens,
                 activation: nn.Module | None = None,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.condition_dim = condition_dim
        self.timestep_dim = timestep_dim

        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.model = DiT(
            d_model=d_model,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim + timestep_dim,
            condition_tokens=condition_tokens,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            is_causal=False,
        )

        self.action_proj = nn.Linear(action_dim, d_model)
        self.out_proj = nn.Linear(d_model, action_dim)
        
    def forward(self, action, condition, timestep_embed):
        return self.out_proj(self.model(self.action_proj(action).unsqueeze(-2), 
                                        torch.cat([condition, timestep_embed], dim=-1)).squeeze(-2))