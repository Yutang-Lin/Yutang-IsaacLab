"""Pure tensor helpers for gamma-conditioned FB and stochastic integration."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def normalized_forward_value(
    q: torch.Tensor,
    gamma: torch.Tensor,
    output_is_normalized: bool,
) -> torch.Tensor:
    """Return (1-gamma)Q, avoiding a redundant scale for normalized F output."""
    if output_is_normalized:
        return q
    return (1.0 - gamma) * q


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


def sample_discrete_gamma(
    reference: torch.Tensor,
    gamma_values: tuple[float, ...],
) -> torch.Tensor:
    """Sample uniformly from a configured finite set of discount factors."""
    if not gamma_values:
        raise ValueError("gamma_values must be non-empty")
    if any(not 0.0 <= gamma < 1.0 for gamma in gamma_values):
        raise ValueError("all gamma_values must be in [0, 1)")
    values = reference.new_tensor(gamma_values)
    indices = torch.randint(
        values.numel(), reference.shape, device=reference.device
    )
    return values[indices]


def scaled_two_gamma_logsumexp(
    normalized_q: torch.Tensor,
    scale: float,
    beta: float = 0.0,
) -> torch.Tensor:
    """Aggregate beta-weighted short/long normalized Q values."""
    if normalized_q.shape[-1] != 2:
        raise ValueError("normalized_q must have exactly two values per row")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if not -1.0 < beta < 1.0:
        raise ValueError("beta must be in (-1, 1)")
    coefficients = normalized_q.new_tensor((1.0 + beta, 1.0 - beta))
    weighted_q = normalized_q * coefficients
    return (scale / math.log(2.0)) * torch.logsumexp(weighted_q, dim=-1)


def horizon_beta_coefficients(
    log_horizons: torch.Tensor,
    h_min: float,
    h_max: float,
    beta: float,
) -> torch.Tensor:
    """Interpolate from 1+beta to 1-beta over log effective horizon."""
    if not h_min < h_max:
        raise ValueError("h_min must be less than h_max")
    if not -1.0 < beta < 1.0:
        raise ValueError("beta must be in (-1, 1)")
    progress = (log_horizons - h_min) / (h_max - h_min)
    return 1.0 + beta - 2.0 * beta * progress


def scaled_horizon_logsumexp(
    normalized_q: torch.Tensor,
    log_horizons: torch.Tensor,
    h_min: float,
    h_max: float,
    scale: float,
    beta: float,
) -> torch.Tensor:
    """Aggregate horizon-conditioned Q with a linear beta schedule."""
    if normalized_q.shape != log_horizons.shape:
        raise ValueError("normalized_q and log_horizons must have equal shape")
    if normalized_q.ndim < 1 or normalized_q.shape[-1] < 2:
        raise ValueError("at least two horizon values are required per row")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    coefficients = horizon_beta_coefficients(
        log_horizons, h_min, h_max, beta
    )
    coefficient_sum = coefficients.sum(dim=-1)
    return (
        scale
        / coefficient_sum.log()
        * torch.logsumexp(coefficients * normalized_q, dim=-1)
    )


def sample_relabel_z(
    stored_z: torch.Tensor,
    mixed_z: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independently relabel rows and return the sampled column mask."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Expected relabel ratio in [0, 1], got {ratio}")
    if stored_z.shape != mixed_z.shape:
        raise ValueError(
            "Stored and mixed z must have matching shapes, got "
            f"{tuple(stored_z.shape)} and {tuple(mixed_z.shape)}"
        )
    mask_shape = (stored_z.shape[0],) + (1,) * (stored_z.ndim - 1)
    if ratio == 0.0:
        mask = torch.zeros(mask_shape, device=stored_z.device, dtype=torch.bool)
    elif ratio == 1.0:
        mask = torch.ones(mask_shape, device=stored_z.device, dtype=torch.bool)
    else:
        mask = torch.rand(mask_shape, device=stored_z.device) < ratio
    return torch.where(mask, mixed_z, stored_z), mask


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
    innovation_target: torch.Tensor,
) -> torch.Tensor:
    """Align one Bellman innovation to a stop-gradient second discount."""
    return 0.5 * F.mse_loss(innovation, innovation_target.detach())


def aux_reward_for_critic(
    raw_reward: torch.Tensor,
    ema_normalized_reward: torch.Tensor,
    fixed_scale: float,
    reward_variance: torch.Tensor | None = None,
    sigma_min: float = 0.0,
) -> torch.Tensor:
    """Normalize aux rewards with a fixed scale or a floored EMA sigma."""
    if fixed_scale < 0.0:
        raise ValueError("fixed_scale must be non-negative")
    if sigma_min < 0.0:
        raise ValueError("sigma_min must be non-negative")
    if fixed_scale > 0.0:
        return raw_reward / fixed_scale
    if sigma_min > 0.0:
        if reward_variance is None:
            raise ValueError(
                "reward_variance is required when sigma_min is positive"
            )
        sigma = reward_variance.clamp_min(0.0).sqrt().detach()
        sigma = sigma.to(device=raw_reward.device, dtype=raw_reward.dtype)
        divisor = sigma.clamp_min(sigma_min)
        return ema_normalized_reward * (sigma / divisor)
    return ema_normalized_reward


def aux_q_for_actor(
    q_aux: torch.Tensor,
    reward_variance: torch.Tensor,
    denormalize: bool,
    fixed_scale: float = 0.0,
    sigma_min: float = 0.0,
) -> torch.Tensor:
    """Apply the configured detached aux-Q reward-scale correction."""
    if not denormalize:
        return q_aux
    if fixed_scale < 0.0:
        raise ValueError("fixed_scale must be non-negative")
    if sigma_min < 0.0:
        raise ValueError("sigma_min must be non-negative")
    if fixed_scale > 0.0:
        scale = q_aux.new_tensor(fixed_scale)
    elif sigma_min > 0.0:
        return q_aux
    else:
        scale = reward_variance.clamp_min(0.0).sqrt().detach()
        scale = scale.to(device=q_aux.device, dtype=q_aux.dtype)
    return q_aux * scale


def tracking_failure_metrics(
    live_joint_pos: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    live_priv: torch.Tensor,
    ref_priv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical-joint MAE and absolute pelvis-height error."""
    if (
        live_joint_pos.shape != ref_joint_pos.shape
        or live_joint_pos.ndim != 2
    ):
        raise ValueError(
            "live_joint_pos and ref_joint_pos must have the same "
            "[batch, joints] shape"
        )
    if live_priv.shape != ref_priv.shape or live_priv.ndim != 2:
        raise ValueError(
            "live_priv and ref_priv must have the same [batch, features] shape"
        )
    if live_joint_pos.shape[0] != live_priv.shape[0]:
        raise ValueError(
            "joint-position and privileged-state batches must match"
        )
    joint_mae = (live_joint_pos - ref_joint_pos).abs().mean(dim=-1)
    root_height_error = (live_priv[:, 0] - ref_priv[:, 0]).abs()
    return joint_mae, root_height_error


def completed_tracking_bins(
    poststep_frames: torch.Tensor,
    bin_ends: torch.Tensor,
    final_frames: torch.Tensor,
) -> torch.Tensor:
    """Return bins whose final transition/frame has been reached."""
    if (
        poststep_frames.shape != bin_ends.shape
        or poststep_frames.shape != final_frames.shape
    ):
        raise ValueError("tracking bin tensors must have matching shapes")
    return poststep_frames >= torch.minimum(bin_ends, final_frames)


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
