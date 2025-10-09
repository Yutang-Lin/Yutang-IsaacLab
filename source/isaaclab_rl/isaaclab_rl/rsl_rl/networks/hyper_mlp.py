import torch
import torch.nn as nn
from .hyper_layer import HyperLayer
from ..utils import TensorDict

class HyperMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, control_dim,
                 hyper_layer_idx):
        super(HyperMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(approximate="tanh"),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[0], output_dim)
        )
        self.control_proj = nn.Sequential(
            nn.Linear(control_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dims[0], 128),
        )
        self.hyper_layers = nn.ModuleList([HyperLayer(hidden_dims[0], 128) for i in range(len(hidden_dims)-1)])

    def orthogonal_loss(self):
        loss = 0.0
        for layer in self.hyper_layers:
            loss += layer.orthogonal_loss()
        return loss

    def forward(self, x: TensorDict):
        proprio = x['proprio']
        control = x['control']
        control = self.control_proj(control)
        proprio = self.input_proj(proprio)
        for layer in self.hyper_layers:
            proprio = layer(proprio, control)
        return self.output_proj(proprio)