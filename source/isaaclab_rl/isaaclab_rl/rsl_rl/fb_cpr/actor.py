import torch
import torch.nn as nn
from ..networks.transformer_policy import TransformerPolicy

class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action, dim_latent):
        super(Actor, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.dim_latent = dim_latent

        self.actor = TransformerPolicy(
            input_size=dim_observation + dim_latent,
            output_size=dim_action,
            mlp_hidden_dims=[256],
            mlp_activation=nn.Mish(),
            num_input_tokens=2,
            d_model=256,
            num_layers=2,
            num_heads=4,
            hidden_dim=512,
            dropout=0.0,
            activation=nn.GELU(approximate="tanh"),
        )

    def forward(self, observation, latent):
        inputs = torch.cat([observation, latent], dim=-1)
        return self.actor(inputs)