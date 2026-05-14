# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""D435i depth noise simulation — GPU-batched PyTorch version for training.

Applies realistic stereo-camera noise during DAgger rollouts so the
student learns to handle real D435i depth artifacts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class D435iDepthNoise:
    """GPU-batched D435i noise model for training.

    Applied to [N, H, W] depth tensors on the same device as the env.
    Stateless (no temporal correlation) for training simplicity — each
    step is independent, which provides stronger augmentation.

    Noise model:
      1. Range gate: [z_min, z_max], out-of-range → 0
      2. z^2 Gaussian noise (stereo disparity error)
      3. Random pixel dropout (distance-dependent)
      4. Edge dropout (depth gradient-based)
      5. Millimeter quantization
    """

    def __init__(
        self,
        z_min: float = 0.3,
        z_max: float = 3.0,
        alpha: float = 0.005,
        beta: float = 0.001,
        hole_base: float = 0.001,
        far_hole: float = 0.02,
        edge_hole: float = 0.4,
        edge_grad_threshold: float = 0.06,
        quant_mm: float = 1.0,
    ):
        self.z_min = z_min
        self.z_max = z_max
        self.alpha = alpha
        self.beta = beta
        self.hole_base = hole_base
        self.far_hole = far_hole
        self.edge_hole = edge_hole
        self.edge_grad_threshold = edge_grad_threshold
        self.quant = quant_mm * 0.001

    @torch.no_grad()
    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply noise to batched depth.

        Args:
            depth: [N, H, W] clean depth (meters). 0 = no measurement.

        Returns:
            [N, H, W] noisy depth. 0 = invalid.
        """
        z = depth.clone()
        N, H, W = z.shape
        device = z.device

        valid = z > 0.01

        # Range gate
        invalid = (~valid) | (z < self.z_min) | (z > self.z_max)

        # z^2 Gaussian noise
        sigma = self.alpha * z * z + self.beta
        noise = torch.randn_like(z) * sigma
        z = z + noise

        # Edge detection via Sobel-like gradient
        # Use 1D finite differences for speed (no conv2d kernel needed)
        z_for_grad = torch.where(valid, z, torch.zeros_like(z))
        dx = torch.zeros_like(z)
        dy = torch.zeros_like(z)
        dx[:, :, 1:] = z_for_grad[:, :, 1:] - z_for_grad[:, :, :-1]
        dy[:, 1:, :] = z_for_grad[:, 1:, :] - z_for_grad[:, :-1, :]
        grad = (dx.abs() + dy.abs())
        edge = grad > self.edge_grad_threshold

        # Distance-dependent dropout
        far_factor = ((z - 1.0) / max(self.z_max - 1.0, 1e-6)).clamp(0, 1)
        p_hole = self.hole_base + self.far_hole * far_factor * far_factor
        p_hole = torch.where(edge, p_hole + self.edge_hole, p_hole)
        holes = torch.rand(N, H, W, device=device) < p_hole
        invalid = invalid | holes

        # Quantization
        z = torch.round(z / self.quant) * self.quant

        # Apply invalids
        z[invalid] = 0.0
        z[z < self.z_min] = 0.0
        z[z > self.z_max] = 0.0

        return z
