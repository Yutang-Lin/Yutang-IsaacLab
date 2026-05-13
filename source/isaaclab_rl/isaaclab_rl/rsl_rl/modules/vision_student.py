# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Vision-based student policy for BFM DAgger distillation.

Replaces height-scan with a depth image encoder (lightweight CNN).
Receives z from teacher's backward map to preserve multi-behavior capability.
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


class VisionStudent(nn.Module):
    """Vision DAgger student: depth CNN + (proprio, z) → action MLP."""

    is_recurrent = False

    def __init__(
        self,
        num_proprio: int,
        num_actions: int,
        z_dim: int,
        depth_height: int = 58,
        depth_width: int = 87,
        depth_feature_dim: int = 128,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
    ):
        super().__init__()
        self.num_proprio = num_proprio
        self.num_actions = num_actions
        self.z_dim = z_dim
        self.depth_height = depth_height
        self.depth_width = depth_width

        self.depth_encoder = DepthEncoder(depth_height, depth_width, depth_feature_dim)

        mlp_input_dim = depth_feature_dim + num_proprio + z_dim
        act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[activation]

        layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, depth: torch.Tensor, proprio: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        depth_feat = self.depth_encoder(depth)
        x = torch.cat([depth_feat, proprio, z], dim=-1)
        return self.mlp(x)

    def act_inference(self, depth: torch.Tensor, proprio: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.forward(depth, proprio, z)
