import os
os.environ["ENABLE_ISAACLAB"] = "False"

import copy
import importlib.util
import math
import sys
from pathlib import Path

import gymnasium
import pytest
import torch

from isaaclab_rl.rsl_rl.fb_cpr.fb_networks import test_fb_networks
from isaaclab_rl.rsl_rl.fb_cpr_math import normalized_horizon_to_gamma

_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_rl/rsl_rl/modules/fb_cpr_policy.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_fb_cpr_policy_module", _POLICY_PATH
)
_POLICY_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _POLICY_MODULE
_SPEC.loader.exec_module(_POLICY_MODULE)
FourierMLP = _POLICY_MODULE.FourierMLP
ForwardMap = _POLICY_MODULE.ForwardMap
Actor = _POLICY_MODULE.Actor
ScalarMLP = _POLICY_MODULE.ScalarMLP
gamma_forward_output_to_raw = _POLICY_MODULE.gamma_forward_output_to_raw
weight_init = _POLICY_MODULE.weight_init


def _actor(predict_horizon: bool) -> Actor:
    return Actor(
        gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
        z_dim=5,
        action_dim=3,
        hidden_dim=16,
        model="simple",
        hidden_layers=1,
        embedding_layers=2,
        predict_horizon=predict_horizon,
    )


def test_actor_legacy_output_and_state_dict_are_unchanged():
    actor = _actor(predict_horizon=False)
    obs = torch.randn(4, 7)
    z = torch.randn(4, 5)

    dist = actor(obs, z, 0.05)

    assert dist.mean.shape == (4, 3)
    assert not any(key.startswith("horizon_head.") for key in actor.state_dict())


def test_actor_horizon_head_starts_at_midpoint_and_receives_gradient():
    actor = _actor(predict_horizon=True)
    obs = torch.randn(4, 7)
    z = torch.randn(4, 5)

    _, normalized_horizon = actor(
        obs, z, 0.05, return_horizon=True
    )

    torch.testing.assert_close(
        normalized_horizon,
        torch.full_like(normalized_horizon, 0.5),
    )
    normalized_horizon.sum().backward()
    assert actor.horizon_head is not None
    assert actor.horizon_head.weight.grad is not None
    assert actor.horizon_head.weight.grad.abs().sum() > 0.0


def test_actor_horizon_head_uses_final_action_hidden_activation():
    torch.manual_seed(11)
    actor = _actor(predict_horizon=True)
    assert actor.horizon_head is not None
    with torch.no_grad():
        actor.horizon_head.weight.normal_()
        actor.horizon_head.bias.normal_()
    obs = torch.randn(4, 7)
    z = torch.randn(4, 5)

    _, normalized_horizon = actor(
        obs, z, 0.05, return_horizon=True
    )
    z_embedding = actor.embed_z(torch.cat([obs, z], dim=-1))
    s_embedding = actor.embed_s(obs)
    hidden = torch.cat([s_embedding, z_embedding], dim=-1)
    for layer in list(actor.policy.children())[:-1]:
        hidden = layer(hidden)
    expected = torch.sigmoid(actor.horizon_head(hidden))

    torch.testing.assert_close(normalized_horizon, expected)


def test_actor_action_and_horizon_share_one_trunk_forward():
    actor = _actor(predict_horizon=True)
    call_counts = [0] * len(actor.policy)
    handles = []
    for index, layer in enumerate(actor.policy):
        def _count_call(_module, _inputs, _output, index=index):
            call_counts[index] += 1

        handles.append(layer.register_forward_hook(_count_call))
    try:
        actor(
            torch.randn(4, 7),
            torch.randn(4, 5),
            0.05,
            return_horizon=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert call_counts == [1] * len(actor.policy)


def test_actor_rejects_horizon_output_when_head_is_disabled():
    actor = _actor(predict_horizon=False)
    with pytest.raises(RuntimeError, match="predict_horizon=True"):
        actor(
            torch.randn(2, 7),
            torch.randn(2, 5),
            0.05,
            return_horizon=True,
        )


def test_raw_fb_objective_reaches_action_and_horizon_heads():
    torch.manual_seed(17)
    actor = _actor(predict_horizon=True)
    forward_map = _gamma_forward_map("mlp")
    obs = torch.randn(8, 7)
    z = torch.randn(8, 5)

    dist, normalized_horizon = actor(
        obs, z, 0.05, return_horizon=True
    )
    gamma = normalized_horizon_to_gamma(
        normalized_horizon, 0.6, 0.99
    ).reshape(-1)
    fs = forward_map(obs, z, dist.mean, gamma)
    q_raw = (fs * z).sum(dim=-1).mean(dim=0)
    loss = -q_raw.mean()
    loss.backward()

    assert actor.horizon_head is not None
    assert actor.horizon_head.weight.grad is not None
    assert actor.horizon_head.weight.grad.abs().sum() > 0.0
    action_head = actor.policy[-1]
    assert action_head.weight.grad is not None
    assert action_head.weight.grad.abs().sum() > 0.0


def _gamma_forward_map(
    embed_type: str,
    scale_hidden_dim: int = 0,
) -> ForwardMap:
    return ForwardMap(
        gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
        z_dim=5,
        action_dim=3,
        hidden_dim=16,
        model="simple",
        hidden_layers=1,
        embedding_layers=2,
        num_parallel=2,
        gamma_embed_dim=8,
        gamma_embed_type=embed_type,
        gamma_scale_hidden_dim=scale_hidden_dim,
    )


@pytest.mark.parametrize(
    ("embed_type", "embed_class", "first_input_dim"),
    [("fourier", FourierMLP, 32), ("mlp", ScalarMLP, 1)],
)
def test_forward_map_gamma_embedding_types(
    embed_type, embed_class, first_input_dim
):
    module = _gamma_forward_map(embed_type)

    assert isinstance(module.embed_gamma, embed_class)
    assert module.embed_gamma.mlp[0].in_features == first_input_dim
    output = module(
        torch.randn(4, 7),
        torch.randn(4, 5),
        torch.randn(4, 3),
        torch.linspace(0.6, 0.99, 4),
    )
    assert output.shape == (2, 4, 5)
    assert torch.isfinite(output).all()


def test_forward_map_rejects_unknown_gamma_embedding_type():
    with pytest.raises(ValueError, match="gamma_embed_type"):
        _gamma_forward_map("unknown")


def test_gamma_scale_shortcut_is_identity_after_policy_initialization():
    module = _gamma_forward_map("mlp", scale_hidden_dim=4)
    module.apply(weight_init)

    gamma = torch.linspace(0.6, 0.99, 5)
    horizon = -torch.log1p(-gamma).unsqueeze(-1)
    gamma_embedding = module.embed_gamma(horizon)
    scale_hidden = torch.nn.functional.mish(
        module.gamma_scale_hidden(gamma_embedding)
    )
    log_scale = module.gamma_scale_output(scale_hidden)
    scale = torch.exp(log_scale).transpose(0, 1).unsqueeze(-1)

    assert scale.shape == (2, 5, 1)
    torch.testing.assert_close(scale, torch.ones_like(scale))


def test_gamma_scale_shortcut_multiplies_raw_f_and_receives_gradient():
    module = _gamma_forward_map("mlp", scale_hidden_dim=4)
    module.apply(weight_init)
    obs = torch.randn(4, 7)
    z = torch.randn(4, 5)
    action = torch.randn(4, 3)
    gamma = torch.linspace(0.6, 0.99, 4)

    identity_output = module(obs, z, action, gamma).detach()
    with torch.no_grad():
        module.gamma_scale_output.bias.copy_(
            torch.tensor([math.log(2.0), math.log(3.0)])
        )
    scaled_output = module(obs, z, action, gamma)

    expected_scale = torch.tensor([2.0, 3.0]).view(2, 1, 1)
    torch.testing.assert_close(scaled_output, expected_scale * identity_output)
    scaled_output.sum().backward()
    assert module.gamma_scale_output.weight.grad is not None
    assert module.gamma_scale_output.weight.grad.abs().sum() > 0


def test_gamma_scale_shortcut_requires_gamma_conditioning():
    with pytest.raises(ValueError, match="requires gamma_embed_dim"):
        ForwardMap(
            gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
            z_dim=5,
            action_dim=3,
            gamma_scale_hidden_dim=4,
        )


def test_gamma_scale_shortcut_is_owned_by_forward_map_target_copy():
    online = _gamma_forward_map("mlp", scale_hidden_dim=4)
    online.apply(weight_init)
    target = copy.deepcopy(online)

    online_names = dict(online.named_parameters())
    target_names = dict(target.named_parameters())
    for name in (
        "gamma_scale_hidden.weight",
        "gamma_scale_hidden.bias",
        "gamma_scale_output.weight",
        "gamma_scale_output.bias",
    ):
        assert name in online_names
        assert name in target_names
        assert online_names[name] is not target_names[name]
        torch.testing.assert_close(online_names[name], target_names[name])


def test_normalized_gamma_output_reconstructs_raw_f():
    output = torch.ones(2, 2, 3, requires_grad=True)
    gamma = torch.tensor([0.6, 0.99])

    raw = gamma_forward_output_to_raw(output, gamma)

    torch.testing.assert_close(raw[:, 0], torch.full((2, 3), 2.5))
    torch.testing.assert_close(raw[:, 1], torch.full((2, 3), 100.0))
    raw.sum().backward()
    torch.testing.assert_close(
        output.grad[:, 0], torch.full((2, 3), 2.5)
    )
    torch.testing.assert_close(
        output.grad[:, 1], torch.full((2, 3), 100.0)
    )


if __name__ == "__main__":
    test_fb_networks()
