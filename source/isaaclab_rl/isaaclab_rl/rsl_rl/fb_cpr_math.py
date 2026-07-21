"""Pure tensor helpers for gamma-conditioned FB and stochastic integration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def centered_context_offsets(
    mean_widths: torch.Tensor,
    context_width: int,
) -> torch.Tensor:
    """Return context starts relative to first-T means with aligned midpoints."""
    return torch.div(
        mean_widths - context_width,
        2,
        rounding_mode="floor",
    )


def centered_subwindow_start(
    container_length: int,
    subwindow_length: int,
) -> int:
    """Return the left-biased centered start for a fixed-width subwindow."""
    return (container_length - subwindow_length + 1) // 2


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
    power: float = 2.0,
) -> torch.Tensor:
    """Return ``(1-gamma)^power`` with unit expectation under log-h sampling."""
    if not 0.0 <= gamma_min < gamma_max < 1.0:
        raise ValueError(
            f"Expected 0 <= gamma_min < gamma_max < 1, got {gamma_min}, {gamma_max}"
        )
    if power < 0.0:
        raise ValueError(f"Expected power >= 0, got {power}")
    if power == 0.0:
        return torch.ones_like(gamma)
    h_min = -math.log1p(-gamma_min)
    h_max = -math.log1p(-gamma_max)
    expected_weight = (
        (1.0 - gamma_min) ** power - (1.0 - gamma_max) ** power
    ) / (power * (h_max - h_min))
    return (1.0 - gamma).pow(power) / expected_weight


def innovation_alignment_loss(
    innovation: torch.Tensor,
    innovation_alt: torch.Tensor,
) -> torch.Tensor:
    """Return half-MSE between Bellman innovations at two discounts."""
    return 0.5 * F.mse_loss(innovation, innovation_alt)


def aux_q_for_actor(
    q_aux: torch.Tensor,
    reward_variance: torch.Tensor,
    denormalize: bool,
) -> torch.Tensor:
    """Optionally restore normalized Q_aux to detached reward-scale units."""
    if not denormalize:
        return q_aux
    sigma = reward_variance.clamp_min(0.0).sqrt().detach()
    return q_aux * sigma.to(device=q_aux.device, dtype=q_aux.dtype)


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
