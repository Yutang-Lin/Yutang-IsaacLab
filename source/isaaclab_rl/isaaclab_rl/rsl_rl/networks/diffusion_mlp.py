import torch
import torch.nn as nn

class DiffusionMLP(nn.Module):
    def __init__(self, action_dim, 
                 condition_dim,
                 hidden_dims, 
                 condition_hidden_dim,
                 timestep_hidden_dim,
                 activation: nn.Module):
        super().__init__()
        self.condition_dim = condition_dim
        if condition_dim > 0:
            self.condition_proj = nn.Sequential(
                nn.Linear(condition_dim, condition_hidden_dim),
                activation,
                nn.Linear(condition_hidden_dim, condition_hidden_dim),
            )
        else:
            self.condition_proj = None
        self.in_features = condition_dim

        mlp_input_dim = action_dim + condition_hidden_dim + timestep_hidden_dim
        hidden_dims = [mlp_input_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, action, condition, timestep_embed):
        if self.condition_dim > 0 and self.condition_proj is not None:
            condition = self.condition_proj(condition)
        mlp_input = torch.cat([action, condition, timestep_embed], dim=-1)
        return self.mlp(mlp_input)
        
        