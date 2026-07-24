import math
import os

os.environ["ENABLE_ISAACLAB"] = "False"

import torch

from isaaclab_rl.rsl_rl.fb_cpr_math import (
    advance_tracking_phases,
    aux_q_for_actor,
    aux_reward_for_critic,
    centered_context_offsets,
    centered_subwindow_start,
    completed_tracking_bins,
    innovation_alignment_loss,
    normalized_gamma_loss_weights,
    sample_log_horizon_gamma,
    stochastic_integral_weights,
    tracking_failure_metrics,
)


def test_aux_critic_uses_fixed_reward_scale_instead_of_ema():
    raw = torch.tensor([[-128.0], [64.0]])
    adaptive = torch.tensor([[-1.0], [0.5]])

    output = aux_reward_for_critic(raw, adaptive, fixed_scale=64.0)

    torch.testing.assert_close(output, torch.tensor([[-2.0], [1.0]]))
    torch.testing.assert_close(
        aux_reward_for_critic(raw, adaptive, fixed_scale=0.0),
        adaptive,
    )


def test_aux_actor_q_is_unchanged_by_default():
    q_aux = torch.tensor([1.0, 3.0], requires_grad=True)
    output = aux_q_for_actor(q_aux, torch.tensor(4.0), denormalize=False)

    torch.testing.assert_close(output, q_aux)


def test_aux_actor_q_restores_detached_reward_sigma():
    q_aux = torch.tensor([1.0, 3.0], requires_grad=True)
    variance = torch.tensor(4.0, requires_grad=True)
    output = aux_q_for_actor(q_aux, variance, denormalize=True)

    torch.testing.assert_close(output, 2.0 * q_aux)
    output.sum().backward()
    torch.testing.assert_close(q_aux.grad, torch.full_like(q_aux, 2.0))
    assert variance.grad is None


def test_aux_actor_q_uses_fixed_scale_instead_of_ema_sigma():
    q_aux = torch.tensor([1.0, 3.0], requires_grad=True)
    output = aux_q_for_actor(
        q_aux,
        reward_variance=torch.tensor(4.0),
        denormalize=True,
        fixed_scale=64.0,
    )

    torch.testing.assert_close(output, 64.0 * q_aux)
    output.sum().backward()
    torch.testing.assert_close(q_aux.grad, torch.full_like(q_aux, 64.0))


def test_tracking_failure_metric_uses_canonical_joint_mae():
    ref_joint_pos = torch.zeros(2, 29)
    live_joint_pos = ref_joint_pos.clone()
    live_joint_pos[0] = 0.3
    ref = torch.zeros(2, 463)
    live = ref.clone()
    live[1, 0] = 0.25

    joint_mae, root_h_error = tracking_failure_metrics(
        live_joint_pos, ref_joint_pos, live, ref
    )

    torch.testing.assert_close(joint_mae, torch.tensor([0.3, 0.0]))
    torch.testing.assert_close(root_h_error, torch.tensor([0.0, 0.25]))


def test_tracking_bins_complete_at_next_bin_start_or_motion_end():
    completed = completed_tracking_bins(
        poststep_frames=torch.tensor([49, 50, 118, 119]),
        bin_ends=torch.tensor([50, 50, 120, 120]),
        final_frames=torch.tensor([119, 119, 119, 119]),
    )

    assert completed.tolist() == [False, True, False, True]


def test_tracking_phase_holds_reset_slot_while_other_slots_advance():
    phases = torch.tensor([11, 20, 249])
    hold = torch.tensor([True, False, False])

    advanced = advance_tracking_phases(phases, hold, max_phase=249)

    assert advanced.tolist() == [11, 21, 249]


def test_centered_positive_context_preserves_first_t_mean():
    widths = torch.tensor([1, 2, 4, 8, 16])
    offsets = centered_context_offsets(widths, context_width=16)

    assert offsets.tolist() == [-8, -7, -6, -4, 0]
    mean_center_twice = widths - 1
    context_center_twice = 2 * offsets + 16 - 1
    assert (mean_center_twice - context_center_twice).tolist() == [1, 0, 0, 0, 0]


def test_configurable_positive_window_is_centered_in_context():
    assert centered_subwindow_start(16, 16) == 0
    assert centered_subwindow_start(16, 8) == 4
    assert centered_subwindow_start(16, 1) == 8


def test_independent_gamma_samples_stay_in_range():
    torch.manual_seed(7)
    reference = torch.empty(4096)
    gamma = sample_log_horizon_gamma(reference, 0.4, 0.975)
    gamma_alt = sample_log_horizon_gamma(reference, 0.4, 0.975)

    assert torch.all(gamma >= 0.4)
    assert torch.all(gamma <= 0.975)
    assert torch.all(gamma_alt >= 0.4)
    assert torch.all(gamma_alt <= 0.975)
    assert not torch.equal(gamma, gamma_alt)


def test_gamma99_log_horizon_distribution():
    torch.manual_seed(17)
    reference = torch.empty(200_000)
    gamma = sample_log_horizon_gamma(reference, 0.4, 0.99)

    assert torch.all(gamma >= 0.4)
    assert torch.all(gamma <= 0.99)
    # Uniform log-horizon sampling intentionally puts a substantial, stable
    # fraction of FB rows in the long-horizon tail.
    tail = (gamma > 0.975).float().mean()
    torch.testing.assert_close(tail, torch.tensor(0.2238), atol=0.004, rtol=0)


def test_gamma_loss_weights_have_unit_log_horizon_expectation():
    h_min = -math.log1p(-0.4)
    h_max = -math.log1p(-0.99)
    edges = torch.linspace(h_min, h_max, 100_001, dtype=torch.float64)
    h = 0.5 * (edges[:-1] + edges[1:])
    gamma = 1.0 - torch.exp(-h)
    weights = normalized_gamma_loss_weights(gamma, 0.4, 0.99)

    torch.testing.assert_close(
        weights.mean(), torch.tensor(1.0, dtype=weights.dtype), atol=1e-9, rtol=0
    )
    assert weights[gamma.argmin()] > weights[gamma.argmax()]


def test_soft_gamma_loss_weights_have_unit_log_horizon_expectation():
    h_min = -math.log1p(-0.4)
    h_max = -math.log1p(-0.99)
    edges = torch.linspace(h_min, h_max, 100_001, dtype=torch.float64)
    h = 0.5 * (edges[:-1] + edges[1:])
    gamma = 1.0 - torch.exp(-h)
    weights = normalized_gamma_loss_weights(gamma, 0.4, 0.99, power=1.0)

    torch.testing.assert_close(
        weights.mean(), torch.tensor(1.0, dtype=weights.dtype), atol=1e-9, rtol=0
    )
    assert weights[gamma.argmin()] > weights[gamma.argmax()]


def test_innovation_alignment_is_zero_for_matching_innovations():
    innovation = torch.randn(2, 8, 8)
    loss = innovation_alignment_loss(innovation, innovation.clone())
    torch.testing.assert_close(loss, torch.zeros_like(loss))


def test_zero_prior_preserves_original_softmax():
    values = torch.tensor([[0.1, -0.2, 0.7]])
    horizons = torch.tensor([[0.5, 1.0, 2.0]])
    weights, tau = stochastic_integral_weights(
        values, horizons, h_min=0.5, prior_lambda=0.0, adaptive_temperature=False
    )

    torch.testing.assert_close(weights, torch.softmax(values, dim=1))
    torch.testing.assert_close(tau, torch.ones(1, 1))


def test_positive_prior_prefers_short_horizons_for_equal_values():
    values = torch.zeros(1, 4)
    horizons = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
    weights, _ = stochastic_integral_weights(
        values, horizons, h_min=0.5, prior_lambda=1.0, adaptive_temperature=False
    )

    assert torch.all(weights[:, :-1] > weights[:, 1:])


def test_adaptive_temperature_is_translation_invariant():
    values = torch.tensor([[0.0, 1.0, 4.0], [-3.0, 2.0, 8.0]])
    horizons = torch.tensor([[0.5, 1.0, 2.0], [0.5, 1.0, 2.0]])
    shifted = values + torch.tensor([[100.0], [-57.0]])

    weights, tau = stochastic_integral_weights(
        values, horizons, h_min=0.5, prior_lambda=0.5, adaptive_temperature=True
    )
    shifted_weights, shifted_tau = stochastic_integral_weights(
        shifted, horizons, h_min=0.5, prior_lambda=0.5, adaptive_temperature=True
    )

    torch.testing.assert_close(shifted_tau, tau)
    torch.testing.assert_close(shifted_weights, weights)


def test_adaptive_temperature_uses_sqrt_mean_max_gap_with_unit_floor():
    values = torch.tensor([[3.0, 3.0, 3.0], [-3.0, 0.0, 3.0]])
    horizons = torch.zeros_like(values)

    _, tau = stochastic_integral_weights(
        values, horizons, h_min=0.0, prior_lambda=0.0, adaptive_temperature=True
    )

    torch.testing.assert_close(tau[0], torch.tensor([1.0]))
    torch.testing.assert_close(tau[1], torch.tensor([3.0**0.5]))


def test_adaptive_temperature_allows_sqrt_scale_sharpening():
    values = torch.tensor([[0.0, 2.0, 4.0]])
    horizons = torch.zeros_like(values)

    weights, tau = stochastic_integral_weights(
        values, horizons, h_min=0.0, prior_lambda=0.0, adaptive_temperature=True
    )
    scaled_weights, scaled_tau = stochastic_integral_weights(
        9.0 * values,
        horizons,
        h_min=0.0,
        prior_lambda=0.0,
        adaptive_temperature=True,
    )

    torch.testing.assert_close(scaled_tau, 3.0 * tau)
    assert scaled_weights[0, -1] > weights[0, -1]
