"""Pure tensor helpers for gamma-conditioned FB and stochastic integration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sample_log_horizon_gamma(
    reference: torch.Tensor,
    gamma_min: float,
    gamma_max: float,
) -> torch.Tensor:
    """Sample gamma by drawing h=-log(1-gamma) uniformly over the given range."""
    if not 0.0 <= gamma_min < gamma_max < 1.0:
        raise ValueError(
            f"Expected 0 <= gamma_min < gamma_max < 1, got {gamma_min}, {gamma_max}"
        )
    h_min = -math.log1p(-gamma_min)
    h_max = -math.log1p(-gamma_max)
    h = h_min + torch.rand_like(reference) * (h_max - h_min)
    return 1.0 - torch.exp(-h)


def innovation_alignment_loss(
    innovation: torch.Tensor,
    innovation_alt: torch.Tensor,
) -> torch.Tensor:
    """Return half-MSE between Bellman innovations at two discounts."""
    return 0.5 * F.mse_loss(innovation, innovation_alt)


def stochastic_integral_weights(
    target_values: torch.Tensor,
    horizons: torch.Tensor,
    h_min: float,
    prior_lambda: float,
    adaptive_temperature: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SI softmax weights and the per-row temperature."""
    if prior_lambda < 0.0:
        raise ValueError("fb_integral_prior_lambda must be non-negative")
    logits = target_values - target_values.max(dim=1, keepdim=True).values
    if adaptive_temperature:
        tau = target_values.mean(dim=1, keepdim=True).abs().sqrt().clamp_min(1.0)
        logits = logits / tau
    else:
        tau = torch.ones_like(target_values[:, :1])
    if prior_lambda > 0.0:
        logits = logits - prior_lambda * (horizons - h_min)
    return torch.softmax(logits, dim=1), tau
