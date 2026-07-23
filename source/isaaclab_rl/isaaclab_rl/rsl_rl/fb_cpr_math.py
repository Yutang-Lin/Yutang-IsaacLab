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


def aux_reward_for_critic(
    raw_reward: torch.Tensor,
    ema_normalized_reward: torch.Tensor,
    fixed_scale: float,
) -> torch.Tensor:
    """Use fixed numerical reward units when requested, else adaptive units."""
    if fixed_scale < 0.0:
        raise ValueError("fixed_scale must be non-negative")
    if fixed_scale > 0.0:
        return raw_reward / fixed_scale
    return ema_normalized_reward


def aux_q_for_actor(
    q_aux: torch.Tensor,
    reward_variance: torch.Tensor,
    denormalize: bool,
    fixed_scale: float = 0.0,
) -> torch.Tensor:
    """Optionally restore normalized Q_aux to detached reward-scale units."""
    if not denormalize:
        return q_aux
    if fixed_scale < 0.0:
        raise ValueError("fixed_scale must be non-negative")
    if fixed_scale > 0.0:
        scale = q_aux.new_tensor(fixed_scale)
    else:
        scale = reward_variance.clamp_min(0.0).sqrt().detach()
        scale = scale.to(device=q_aux.device, dtype=q_aux.dtype)
    return q_aux * scale


def tracking_failure_metrics(
    live_priv: torch.Tensor,
    ref_priv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return heading-local keypoint MPJPE and absolute root-height error.

    Supports the standard ``max_local_self`` layout ``15*K-2`` and its
    optional heading-body-tail layout ``24*K-5``.
    """
    if live_priv.shape != ref_priv.shape or live_priv.ndim != 2:
        raise ValueError(
            "live_priv and ref_priv must have the same [batch, features] shape"
        )
    priv_dim = int(ref_priv.shape[-1])
    if (priv_dim + 2) % 15 == 0:
        num_bodies = (priv_dim + 2) // 15
    elif (priv_dim + 5) % 24 == 0:
        num_bodies = (priv_dim + 5) // 24
    else:
        raise ValueError(
            f"Cannot infer max_local_self body count from dim={priv_dim}"
        )
    pos_dim = 3 * (num_bodies - 1)
    live_pos = live_priv[:, 1 : 1 + pos_dim].view(
        -1, num_bodies - 1, 3
    )
    ref_pos = ref_priv[:, 1 : 1 + pos_dim].view_as(live_pos)
    local_mpjpe = torch.linalg.vector_norm(
        live_pos - ref_pos, dim=-1
    ).mean(dim=-1)
    root_height_error = (live_priv[:, 0] - ref_priv[:, 0]).abs()
    return local_mpjpe, root_height_error


def tracking_rollback_offsets(
    local_time: torch.Tensor,
    rollback_steps: int,
) -> torch.Tensor:
    """Return non-negative reference offsets after a failure rollback."""
    if rollback_steps < 0:
        raise ValueError("rollback_steps must be non-negative")
    # local_time indexes the z used for the action that just completed; its
    # synchronized post-step reference is local_time + 1.
    return (local_time + 1 - rollback_steps).clamp_min(0)


def advance_tracking_phases(
    local_phases: torch.Tensor,
    hold_once: torch.Tensor,
    max_phase: int,
) -> torch.Tensor:
    """Advance normal tracking slots while holding freshly reset slots once."""
    if local_phases.shape != hold_once.shape:
        raise ValueError("local_phases and hold_once must have matching shapes")
    if max_phase < 0:
        raise ValueError("max_phase must be non-negative")
    return torch.where(
        hold_once, local_phases, local_phases + 1
    ).clamp(max=max_phase)


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
