"""Pure tensor helpers for gamma-conditioned FB and stochastic integration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ema_grad_spike_state(
    grad_norm: float,
    ema: float,
    steps: int,
    decay: float,
    multiplier: float,
    warmup_steps: int,
) -> tuple[float, float, bool]:
    """Advance a winsorized grad-norm EMA and return its pre-update spike test."""
    baseline = grad_norm if steps == 0 or ema <= 1e-12 else ema
    threshold = max(multiplier * baseline, 1e-12)
    spike = steps >= warmup_steps and grad_norm > threshold
    if steps == 0:
        next_ema = grad_norm
    else:
        next_ema = decay * baseline + (1.0 - decay) * min(
            grad_norm, threshold
        )
    return next_ema, threshold, spike


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


def normalized_gamma_loss_weights(
    gamma: torch.Tensor,
    gamma_min: float,
    gamma_max: float,
) -> torch.Tensor:
    """Return ``(1-gamma)^2`` weights with unit expectation under log-h sampling."""
    if not 0.0 <= gamma_min < gamma_max < 1.0:
        raise ValueError(
            f"Expected 0 <= gamma_min < gamma_max < 1, got {gamma_min}, {gamma_max}"
        )
    h_min = -math.log1p(-gamma_min)
    h_max = -math.log1p(-gamma_max)
    expected_square = (
        (1.0 - gamma_min) ** 2 - (1.0 - gamma_max) ** 2
    ) / (2.0 * (h_max - h_min))
    return (1.0 - gamma).square() / expected_square


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
        mean_gap = logits.abs().mean(dim=1, keepdim=True)
        tau = mean_gap.sqrt().clamp_min(1.0)
        logits = logits / tau
    else:
        tau = torch.ones_like(target_values[:, :1])
    if prior_lambda > 0.0:
        logits = logits - prior_lambda * (horizons - h_min)
    return torch.softmax(logits, dim=1), tau
