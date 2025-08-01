import torch
import torch.nn as nn
from .transformer import MultiHeadAttention

class DiTBlock(nn.Module):
    """Context implementation of DiT block"""
    def __init__(self, d_model, 
                 num_heads, 
                 hidden_dim,
                 condition_dim,
                 condition_tokens,
                 dropout=0.0, 
                 is_causal=False,
                 activation: nn.Module | None = None):
        super().__init__()
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.d_model = d_model
        self.condition_tokens = condition_tokens
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, is_causal=is_causal)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            activation,
            nn.Linear(hidden_dim, d_model),
        )
        self.condition_proj = nn.Linear(condition_dim, condition_tokens * d_model)
    
    def forward(self, feature, condition, attn_mask=None):
        inputs = torch.cat([feature, self.condition_proj(condition).view(*condition.shape[:-1], 
                                                                          self.condition_tokens, 
                                                                          self.d_model)], dim=-2)
        if attn_mask is not None:
            raise NotImplementedError("Attn mask not implemented for DiTBlock")
        
        out = self.norm1(self.self_attn(inputs, inputs, attn_mask)[..., :-self.condition_tokens, :] + feature)
        out = self.norm2(self.mlp(out) + out)
        return out, condition
        
class DiT(nn.Module):
    def __init__(self, d_model, 
                 num_heads, 
                 hidden_dim, 
                 condition_dim,
                 condition_tokens,
                 num_layers, 
                 dropout=0.0, 
                 is_causal=False,
                 activation: nn.Module | None = None):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model,
                num_heads,
                hidden_dim,
                condition_dim,
                condition_tokens,
                dropout,
                is_causal,
                activation
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, feature, condition, attn_mask=None):
        for block in self.blocks:
            feature, condition = block(feature, condition, attn_mask)
        return feature
        
        