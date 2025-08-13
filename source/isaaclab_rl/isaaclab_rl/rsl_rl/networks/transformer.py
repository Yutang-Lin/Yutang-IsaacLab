import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend

class MoEFFN(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts, top_k, activation: nn.Module | None = None):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                activation,
                nn.Linear(hidden_dim, d_model)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        xshape = x.shape

        logits = self.router(x)
        chosen_experts = torch.topk(logits, self.top_k, dim=-1)
        chosen_experts, weight = chosen_experts.indices, chosen_experts.values
        weight = weight.softmax(dim=-1)

        results = torch.zeros(*xshape[:-1], self.d_model, device=x.device, dtype=x.dtype)
        for i, expert in enumerate(self.experts):
            chosen_mask = chosen_experts == i
            chosen_batch = torch.any(chosen_experts == i, dim=-1).nonzero()

            if chosen_batch.numel() > 0:
                chosen_weight = (weight * chosen_mask).sum(dim=-1, keepdim=True)
                results[chosen_batch] += expert(x[chosen_batch]) * chosen_weight[chosen_batch]

        return results

class MultiHeadAttention(nn.Module):
    """Flash Attention enhanced MHA"""
    def __init__(self, d_model, num_heads, dropout=0.0, 
                 is_causal=False,
                 enable_sdpa: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self.enable_sdpa = enable_sdpa

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, feature: torch.Tensor, 
                other: torch.Tensor, 
                attn_mask: torch.Tensor | None = None):
        is_causal = self.is_causal
        if is_causal and attn_mask is not None:
            is_causal = False

        # for torch script exportable
        assert (feature.ndim == 3 and other.ndim == 3), "feature and other must be 3D tensors, with B, L, D"
        q_shape = feature.shape
        v_shape = other.shape
            
        q = self.q_proj(feature)
        k = self.k_proj(other)
        v = self.v_proj(other)

        batch_size = q_shape[0]
        q_length = q_shape[-2]
        kv_length = v_shape[-2]
        q = q.view(batch_size, q_length, self.num_heads, self.head_dim).transpose(-2, -3)
        k = k.view(batch_size, kv_length, self.num_heads, self.head_dim).transpose(-2, -3)
        v = v.view(batch_size, kv_length, self.num_heads, self.head_dim).transpose(-2, -3)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(-3) # match head dim

        if not self.enable_sdpa:
            q = q.reshape(batch_size * self.num_heads, q_length, self.head_dim)
            k = k.reshape(batch_size * self.num_heads, kv_length, self.head_dim)
            v = v.reshape(batch_size * self.num_heads, kv_length, self.head_dim)
            attn_score = torch.bmm(q, k.transpose(-2, -1))
            if attn_mask is not None:
                attn_mask = - torch.inf * torch.repeat_interleave(attn_mask, 
                                                                  self.num_heads, dim=-3).reshape(-1, q.shape[-2], k.shape[-2])
                attn_score = attn_score + attn_mask
            attn_score = F.softmax(attn_score / math.sqrt(self.head_dim), dim=-1)
            out = torch.bmm(attn_score, v).reshape(batch_size, self.num_heads, q_length, self.head_dim)

        else:
            # let torch decide the best backend
            out = scaled_dot_product_attention(q, k, v, dropout_p=self.dropout,
                                                attn_mask=attn_mask,
                                                is_causal=is_causal)

        out = out.transpose(-2, -3).contiguous().view(batch_size, q_length, self.d_model)
        return self.out_proj(out)
        
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, 
                 num_heads, 
                 hidden_dim,
                 dropout=0.0, 
                 is_causal=True,
                 activation: nn.Module | None = None,
                 **kwargs):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, is_causal=is_causal)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            activation,
            nn.Linear(hidden_dim, d_model),
        )
        
    def forward(self, feature, other, self_attn_mask=None, cross_attn_mask=None):
        out = self.norm1(self.self_attn(feature, feature, self_attn_mask) + feature)
        out = self.norm2(self.cross_attn(out, other, cross_attn_mask) + out)
        out = self.norm3(self.mlp(out) + out)
        return out
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.0, 
                 is_causal=False,
                 activation: nn.Module | None = None,
                 enable_sdpa: bool = True):
        super().__init__()
        if activation is None:
            activation = nn.GELU(approximate="tanh")
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, is_causal=is_causal, enable_sdpa=enable_sdpa)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            activation,
            nn.Linear(hidden_dim, d_model),
        )
    
    def forward(self, feature: torch.Tensor, 
                attn_mask: torch.Tensor | None = None):
        out = self.norm1(self.self_attn(feature, feature, attn_mask) + feature)
        out = self.norm2(self.mlp(out) + out)
        return out
    
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, 
                 num_heads, 
                 hidden_dim, 
                 num_layers, 
                 dropout=0.0, 
                 is_causal=True,
                 activation: nn.Module | None = None):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, hidden_dim, dropout, is_causal, activation)
            for _ in range(num_layers)
        ])

    def forward(self, feature, other, self_attn_mask=None, cross_attn_mask=None):
        for layer in self.layers:
            feature = layer(feature, other, self_attn_mask, cross_attn_mask)
        return feature
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, num_layers, dropout=0.0, 
                 is_causal=False,
                 activation: nn.Module | None = None,
                 enable_sdpa: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, hidden_dim, dropout, is_causal, activation, enable_sdpa)
            for _ in range(num_layers)
        ])

    def forward(self, feature: torch.Tensor, 
                attn_mask: torch.Tensor | None = None, 
                return_all_layers: bool = False):
        features = [] # for torch script exportable
        for layer in self.layers:
            feature = layer(feature, attn_mask)
            if return_all_layers:
                features.append(feature)
        if return_all_layers:
            return torch.stack(features, dim=0)
        else:
            return feature
