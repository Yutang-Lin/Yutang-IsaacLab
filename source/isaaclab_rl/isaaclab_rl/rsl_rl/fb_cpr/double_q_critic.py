import torch
import torch.nn as nn
from ..networks.transformer_policy import TransformerPolicy
import torch.nn.functional as F

class DoubleQCritic(nn.Module):
    def __init__(self, dim_state, dim_action, dim_latent,
                 gamma: float = 0.98):
        super(DoubleQCritic, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_latent = dim_latent
        self.gamma = gamma

        self.q1 = TransformerPolicy(
            input_size=dim_state + dim_action + dim_latent,
            output_size=1,
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
        self.q2 = TransformerPolicy(
            input_size=dim_state + dim_action + dim_latent,
            output_size=1,
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

    def _compute_q_minimum(self, next_state, next_action, latent):
        q1_next = self.q1(next_state, next_action, latent)
        q2_next = self.q2(next_state, next_action, latent)
        return torch.minimum(q1_next, q2_next)

    def loss(self, state, action, reward, latent, next_state, next_action):
        inputs = torch.cat([state, action, latent], dim=-1)
        with torch.no_grad():
            td_target = self._compute_q_minimum(next_state, next_action, latent)
        q1_loss = F.mse_loss(self.q1(inputs), reward + self.gamma * td_target)
        q2_loss = F.mse_loss(self.q2(inputs), reward + self.gamma * td_target)
        return q1_loss + q2_loss
    
    def as_q_function(self, state, action, latent):
        return self._compute_q_minimum(state, action, latent).squeeze()