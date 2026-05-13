# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Vision-based student policy for BFM DAgger distillation.

Matches the teacher's Actor architecture (residual embedding + residual
trunk) but replaces height_scan with a depth CNN encoder. This ensures
the student has the same capacity as the teacher.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """Lightweight CNN encoder for single-channel depth images."""

    def __init__(self, height: int = 58, width: int = 87, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=0),
            nn.ELU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, height, width)
            flat_dim = self.conv(dummy).numel()
        self.fc = nn.Linear(flat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W] or [B, 1, H, W] depth image."""
        if x.ndim == 3:
            x = x.unsqueeze(1)
        feat = self.conv(x).flatten(1)
        return self.fc(feat)


# --- Residual building blocks (mirrors fb_cpr_policy.py) ---

class _ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Mish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class _Block(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: bool) -> None:
        super().__init__()
        seq = [nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim)]
        if activation:
            seq.append(nn.Mish())
        self.mlp = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _residual_embedding(input_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    assert hidden_layers >= 2
    seq: list[nn.Module] = [_Block(input_dim, hidden_dim, True)]
    for _ in range(hidden_layers - 2):
        seq.append(_ResidualBlock(hidden_dim))
    seq.append(_Block(hidden_dim, hidden_dim // 2, True))
    return nn.Sequential(*seq)


class VisionStudent(nn.Module):
    """Vision DAgger student matching teacher Actor architecture.

    Architecture (mirrors fb_cpr_policy.Actor):
      depth → DepthEncoder → depth_feat
      obs = cat(depth_feat, proprio)
      z_emb = residual_embedding(cat(obs, z))  → hidden_dim//2
      s_emb = residual_embedding(obs)           → hidden_dim//2
      trunk_input = cat(s_emb, z_emb)           → hidden_dim
      output = residual_trunk(trunk_input)       → action_dim
    """

    is_recurrent = False

    def __init__(
        self,
        num_proprio: int,
        num_actions: int,
        z_dim: int,
        depth_height: int = 58,
        depth_width: int = 87,
        depth_feature_dim: int = 128,
        hidden_dim: int = 2048,
        hidden_layers: int = 6,
        embedding_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_proprio = num_proprio
        self.num_actions = num_actions
        self.z_dim = z_dim
        self.depth_height = depth_height
        self.depth_width = depth_width

        self.depth_encoder = DepthEncoder(depth_height, depth_width, depth_feature_dim)

        obs_dim = depth_feature_dim + num_proprio

        # Match teacher Actor: two residual embeddings each → hidden_dim//2
        self.embed_z = _residual_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = _residual_embedding(obs_dim, hidden_dim, embedding_layers)

        # Residual trunk at full hidden_dim (cat of two hidden_dim//2 embeddings)
        seq: list[nn.Module] = [_ResidualBlock(hidden_dim) for _ in range(hidden_layers)]
        seq.append(_Block(hidden_dim, num_actions, False))
        self.trunk = nn.Sequential(*seq)

    def forward(self, depth: torch.Tensor, proprio: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        depth_feat = self.depth_encoder(depth)
        obs = torch.cat([depth_feat, proprio], dim=-1)
        z_emb = self.embed_z(torch.cat([obs, z], dim=-1))
        s_emb = self.embed_s(obs)
        h = torch.cat([s_emb, z_emb], dim=-1)
        return self.trunk(h)

    def act_inference(self, depth: torch.Tensor, proprio: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.forward(depth, proprio, z)
