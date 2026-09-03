# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for BFM-0.7 backward masking (group layout, sampling, B/F wiring).

Loads the masking helpers by path and the policy / algorithm modules with Isaac
Lab and gymnasium stubbed, so the real ``BackwardMap`` / ``ForwardMap`` run on
CPU with the mask path exercised end to end.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import os
import sys
import types

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "isaaclab_rl"))


def _stub(name: str, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class _Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        low = np.asarray(low, dtype=dtype)
        high = np.asarray(high, dtype=dtype)
        self.low = low
        self.high = high
        self.shape = tuple(shape) if shape is not None else low.shape
        self.dtype = dtype


class _Dict:
    def __init__(self, spaces):
        self.spaces = dict(spaces)

    def __getitem__(self, key):
        return self.spaces[key]


def _load_modules():
    if "isaaclab" not in sys.modules:
        _stub("isaaclab")
        _stub("isaaclab.utils", configclass=dataclasses.dataclass)
    if "gymnasium" not in sys.modules or not hasattr(sys.modules["gymnasium"].spaces, "Box"):
        spaces = _stub("gymnasium.spaces", Box=_Box, Dict=_Dict, Space=object)
        _stub("gymnasium", spaces=spaces)
    for name, path in (
        ("isaaclab_rl", _ROOT),
        ("isaaclab_rl.rsl_rl", f"{_ROOT}/rsl_rl"),
        ("isaaclab_rl.rsl_rl.modules", f"{_ROOT}/rsl_rl/modules"),
        ("isaaclab_rl.rsl_rl.algorithms", f"{_ROOT}/rsl_rl/algorithms"),
    ):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            sys.modules[name] = m
    masking = importlib.import_module("isaaclab_rl.rsl_rl.fb_cpr_masking")
    policy = importlib.import_module("isaaclab_rl.rsl_rl.modules.fb_cpr_policy")
    algo = importlib.import_module("isaaclab_rl.rsl_rl.algorithms.fb_cpr")
    return masking, policy, algo


masking, policy_mod, algo_mod = _load_modules()

KEY_DIMS = {"state": 64, "privileged_state": 463, "contact_labels": 4}
KEYS = ("state", "privileged_state", "contact_labels")


def _space(keys=KEYS):
    return _Dict({k: _Box(-np.ones(KEY_DIMS[k]), np.ones(KEY_DIMS[k])) for k in keys})


# --------------------------------------------------------------------------- #
# group layout
# --------------------------------------------------------------------------- #
def test_groups_partition_input_with_expected_sizes():
    names, groups = masking.build_backward_mask_groups(KEYS, KEY_DIMS)
    assert names == ("left_arm", "right_arm", "torso", "left_leg", "right_leg", "pelvis", "contacts")
    sizes = [len(g) for g in groups]
    assert sizes == [119, 119, 66, 102, 102, 19, 4]
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(531))
    # spot checks: left-arm joints, pelvis rot6d, contacts tail
    la = set(groups[0])
    assert set(range(15, 22)) <= la and set(range(44, 51)) <= la and set(range(110, 131)) <= la
    assert set(range(64 + 91, 64 + 97)) <= set(groups[5])          # pelvis rot6d
    assert groups[6] == [527, 528, 529, 530]


def test_groups_follow_key_order_and_reject_unknown_dims():
    names, groups = masking.build_backward_mask_groups(("state", "privileged_state"), KEY_DIMS)
    assert len(groups[6]) == 0 and sorted(i for g in groups for i in g) == list(range(527))
    try:
        masking.build_backward_mask_groups(("state",), {"state": 60})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        masking.build_backward_mask_groups(("history_actor",), {"history_actor": 837})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_group_expand_matrix_matches_groups():
    _, groups = masking.build_backward_mask_groups(KEYS, KEY_DIMS)
    M = masking.group_expand_matrix(groups, 531)
    assert M.shape == (7, 531) and torch.equal(M.sum(dim=0), torch.ones(531))
    m = torch.tensor([[1, 0, 1, 1, 1, 1, 0]], dtype=torch.float32)
    feat = (m @ M)[0]
    assert feat[groups[1]].sum() == 0 and feat[groups[6]].sum() == 0
    assert feat.sum() == 531 - 119 - 4


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_sample_group_mask_rates_forced_off_and_fallback():
    torch.manual_seed(0)
    m = masking.sample_group_mask(200_000, 7, 0.1, fallback_group=5, device="cpu")
    rate = 1.0 - m.mean(dim=0)
    assert torch.allclose(rate[:5], torch.full((5,), 0.1), atol=0.01)
    assert torch.allclose(rate[6], torch.tensor(0.1), atol=0.01)
    assert bool((m.sum(dim=1) >= 1).all())
    # all-masked rows fall back to pelvis: force it by mask_prob close to 1
    m2 = masking.sample_group_mask(1000, 7, 0.999, fallback_group=5, device="cpu")
    assert bool((m2.sum(dim=1) >= 1).all()) and float(m2[:, 5].mean()) > 0.99
    # forced-off contacts is never visible; fallback still guarantees >= 1 group
    m3 = masking.sample_group_mask(5000, 7, 0.1, fallback_group=5, forced_off_groups=(6,), device="cpu")
    assert float(m3[:, 6].sum()) == 0.0 and bool((m3.sum(dim=1) >= 1).all())
    f = masking.full_mask(3, 7, forced_off_groups=(6,))
    assert f.tolist() == [[1, 1, 1, 1, 1, 1, 0]] * 3


# --------------------------------------------------------------------------- #
# networks
# --------------------------------------------------------------------------- #
def _obs(n):
    torch.manual_seed(1)
    return {k: torch.randn(n, KEY_DIMS[k]) for k in KEYS}


def test_backward_map_masks_groups_and_appends_flags():
    _, groups = masking.build_backward_mask_groups(KEYS, KEY_DIMS)
    B = policy_mod.BackwardMap(_space(), z_dim=8, hidden_dim=16, hidden_layers=1,
                               norm=True, input_keys=KEYS, mask_groups=groups)
    assert B.net[0].in_features == 531 + 7
    obs = _obs(4)
    m_full = torch.ones(4, 7)
    z_default = B(obs)
    z_full = B(obs, m_full)
    assert torch.allclose(z_default, z_full)           # None == all visible
    # masking a group makes B invariant to that group's features
    m = torch.ones(4, 7); m[:, 0] = 0.0                 # hide left arm
    z1 = B(obs, m)
    obs2 = {k: v.clone() for k, v in obs.items()}
    la_state = [i for i in groups[0] if i < 64]
    obs2["state"][:, la_state] += 10.0
    la_priv = [i - 64 for i in groups[0] if 64 <= i < 527]
    obs2["privileged_state"][:, la_priv] -= 10.0
    assert torch.allclose(B(obs2, m), z1)
    assert not torch.allclose(B(obs2, m_full), z_full)  # visible -> changes
    # wrong mask width is rejected
    try:
        B(obs, torch.ones(4, 6))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_forward_map_accepts_mask_and_defaults_to_all_visible():
    F = policy_mod.ForwardMap(_space(("state",)), z_dim=8, action_dim=3, hidden_dim=16, model="simple",
                              hidden_layers=1, embedding_layers=2, num_parallel=2, input_keys=("state",),
                              gamma_embed_dim=4, gamma_embed_type="mlp", mask_dim=7)
    obs = {"state": torch.randn(5, 64)}
    z = torch.randn(5, 8); a = torch.randn(5, 3); g = torch.full((5,), 0.9)
    out_none = F(obs, z, a, g)
    out_full = F(obs, z, a, g, mask=torch.ones(5, 7))
    assert out_none.shape == (2, 5, 8) and torch.allclose(out_none, out_full)
    m = torch.ones(5, 7); m[:, 2] = 0.0
    assert not torch.allclose(F(obs, z, a, g, mask=m), out_full)
    F0 = policy_mod.ForwardMap(_space(("state",)), z_dim=8, action_dim=3, hidden_dim=16, model="simple",
                               hidden_layers=1, embedding_layers=2, num_parallel=2, input_keys=("state",))
    try:
        F0(obs, z, a, mask=torch.ones(5, 7))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# --------------------------------------------------------------------------- #
# algorithm helpers
# --------------------------------------------------------------------------- #
class _FakePolicy:
    mask_group_names = masking.MASK_GROUP_NAMES
    num_mask_groups = 7


def _alg(enabled=True):
    alg = algo_mod.FBCprAux.__new__(algo_mod.FBCprAux)
    alg.cfg = algo_mod.FBCprAuxAlgorithmCfg(backward_masking=enabled, mask_group_prob=0.1)
    alg.device = "cpu"
    alg.policy = _FakePolicy()
    alg._env_mask = None
    return alg


def test_algorithm_mask_helpers():
    alg = _alg()
    torch.manual_seed(0)
    m = alg._sample_mask(1000)
    assert m.shape == (1000, 7) and bool((m.sum(dim=1) >= 1).all())
    me = alg._sample_mask(1000, expert=True)
    assert float(me[:, 6].sum()) == 0.0
    assert alg.expert_encode_mask(2).tolist() == [[1, 1, 1, 1, 1, 1, 0]] * 2
    assert alg._mk(m) == {"mask": m} and alg._mk(None) == {}
    # env masks: drawn for all envs first, then only for reset envs
    step = torch.tensor([0, 0, 0, 0])
    alg._refresh_env_masks(step)
    first = alg._env_mask.clone()
    step = torch.tensor([5, 0, 7, 0])
    torch.manual_seed(123)
    alg._refresh_env_masks(step)
    assert torch.equal(alg._env_mask[[0, 2]], first[[0, 2]])
    # forced-off view of an env mask keeps >= 1 group
    alg._env_mask[:] = 0.0
    alg._env_mask[:, 6] = 1.0                            # contacts-only env mask
    fm = alg._env_mask_for(torch.tensor([0, 1]), num_frames=3, expert=True)
    assert fm.shape == (6, 7) and float(fm[:, 6].sum()) == 0.0 and bool((fm[:, 5] == 1).all())
    # disabled -> all helpers are no-ops
    off = _alg(enabled=False)
    assert off._sample_mask(4) is None and off.expert_encode_mask(4) is None and off._mk(None) == {}
    off._refresh_env_masks(torch.zeros(3, dtype=torch.long))
    assert off._env_mask is None


# --------------------------------------------------------------------------- #
# full policy wiring
# --------------------------------------------------------------------------- #
def _small_policy_cfg(**kw):
    Cfg = policy_mod.FBCprNetworkCfg
    base = dict(
        z_dim=8, backward_hidden_dim=16, backward_hidden_layers=1,
        backward_input_keys=KEYS,
        forward_hidden_dim=16, forward_model="simple", forward_hidden_layers=1,
        forward_embedding_layers=2, forward_num_parallel=2,
        forward_input_keys=("state", "last_action"),
        actor_hidden_dim=16, actor_model="simple", actor_hidden_layers=1, actor_embedding_layers=2,
        actor_input_keys=("state", "last_action"),
        critic_hidden_dim=16, critic_model="simple", critic_hidden_layers=1, critic_embedding_layers=2,
        critic_num_parallel=2, critic_input_keys=("state", "last_action"),
        aux_critic_hidden_dim=16, aux_critic_model="simple", aux_critic_hidden_layers=1,
        aux_critic_embedding_layers=2, aux_critic_num_parallel=2, aux_critic_input_keys=("state", "last_action"),
        discriminator_hidden_dim=16, discriminator_hidden_layers=1, discriminator_input_keys=KEYS,
        obs_normalizer_momentum={"state": 0.01, "privileged_state": 0.01, "contact_labels": 0.01, "last_action": 0.01},
        obs_normalizer_allow_mismatching_keys=True,
        forward_gamma_embed_dim=4, forward_gamma_embed_type="mlp",
    )
    base.update(kw)
    return Cfg(**base)


def _full_space():
    dims = dict(KEY_DIMS, last_action=29)
    return _Dict({k: _Box(-np.ones(d), np.ones(d)) for k, d in dims.items()})


def test_full_policy_wires_mask_groups_into_B_and_F():
    pol = policy_mod.FBCprAuxPolicy(_full_space(), action_dim=29, cfg=_small_policy_cfg(backward_mask_groups=True))
    assert pol.num_mask_groups == 7 and pol.mask_group_names == masking.MASK_GROUP_NAMES
    assert pol._backward_map.num_mask_groups == 7 and pol._forward_map.mask_dim == 7
    obs = dict(_obs(3), last_action=torch.randn(3, 29))
    m = torch.ones(3, 7); m[:, 6] = 0.0
    z_a = pol.backward_map(obs, mask=m)
    z_b = pol.backward_map(obs)
    assert z_a.shape == (3, 8) and not torch.allclose(z_a, z_b)
    out = pol.forward_map(obs, z_a, torch.randn(3, 29), gamma=0.9, mask=m)
    assert out.shape == (2, 3, 8)
    # critics / discriminator are NOT mask-conditioned
    assert pol._critic.mask_dim == 0 and pol._aux_critic.mask_dim == 0


def test_full_policy_without_masking_is_unchanged():
    pol = policy_mod.FBCprAuxPolicy(_full_space(), action_dim=29, cfg=_small_policy_cfg())
    assert pol.num_mask_groups == 0 and pol._forward_map.mask_dim == 0
    assert pol._backward_map.net[0].in_features == 531
    obs = dict(_obs(3), last_action=torch.randn(3, 29))
    assert pol.backward_map(obs).shape == (3, 8)
    assert pol.forward_map(obs, torch.randn(3, 8), torch.randn(3, 29), gamma=0.9).shape == (2, 3, 8)
