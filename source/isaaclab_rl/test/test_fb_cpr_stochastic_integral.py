import os

os.environ["ENABLE_ISAACLAB"] = "False"

import torch

from isaaclab_rl.rsl_rl.fb_cpr_math import (
    innovation_alignment_loss,
    sample_log_horizon_gamma,
    stochastic_integral_weights,
)


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
