import torch
import torch.nn as nn
from .transformer import MultiHeadAttention

class DiTBlock(nn.Module):
    def __init__(self, d_model, 
                 num_heads, 
                 hidden_dim, 
                 num_layers, 
                 dropout=0.0, 
                 is_causal=True,
                 activation: nn.Module | None = None):
        super().__init__()
        
        
        