# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Algorithm-level tests for BFM-0.7 replay "motions" (fake policy / replay).

Loads ``algorithms/fb_cpr.py`` with Isaac Lab and gymnasium stubbed so the
replay-tracking rollout context can run on CPU without simulator dependencies.
"""

from __future__ import annotations

import torch

from _fb_stubs import load as _load_module


fb = _load_module("algorithms.fb_cpr")
FBCprAux = fb.FBCprAux
AlgoCfg = fb.FBCprAuxAlgorithmCfg


class _PolicyCfg:
    backward_input_keys = ("state",)


class FakePolicy:
    """B = identity on 'state' (d=2); project_z = identity; no normalization."""

    seq_length = 16
    cfg = _PolicyCfg()
    z_dim = 2

    def _backward_map(self, obs):
        return obs["state"].clone()

    def project_z(self, z):
        return z

    def _normalize(self, obs):
        return obs

    def sample_z(self, n, device="cpu"):
        return torch.full((n, self.z_dim), 99.0, device=device)


def _make_alg(**cfg_overrides) -> FBCprAux:
    alg = FBCprAux.__new__(FBCprAux)
    alg.cfg = AlgoCfg(**cfg_overrides)
    alg.device = "cpu"
    alg.policy = FakePolicy()
    alg._disc_positive_window = 0
    alg._z_buffer = None
    alg._z_buffer_size = 0
    alg._z_buffer_cursor = 0
    return alg


# --------------------------------------------------------------------------- #
# replay-trajectory rollout tracking
# --------------------------------------------------------------------------- #
class FakeReplay:
    """Windows whose 'state' rows count up from a per-window base."""

    def __init__(self, ready=True):
        self.ready = ready
        self.calls = 0

    def sample_trajectory_windows(self, n, length, keys):
        self.calls += 1
        if not self.ready:
            return None
        base = (torch.arange(n, dtype=torch.float32) * 1000.0).view(n, 1, 1)
        rows = torch.arange(length + 1, dtype=torch.float32).view(1, length + 1, 1)
        st = (base + rows).expand(n, length + 1, 2).contiguous()
        return {"observation": {k: st for k in keys}}


def _rollout_alg(num_envs=10, frac=0.3, L=5):
    return _make_alg(
        rollout_expert_trajectories=False,     # no expert tracking (buffer None)
        rollout_expert_trajectories_percentage=0.5,
        rollout_expert_trajectories_length=L,
        rollout_replay_trajectories_percentage=frac,
        update_z_every_step=1000,
        use_mix_rollout=False,
        tracking_T_min=2, tracking_T_max=2,
    )


def test_replay_tracking_excludes_expert_tracking_envs():
    alg = _rollout_alg(num_envs=10, frac=0.5)
    # Expert tracking owns 7 of 10 envs -> only 3 candidates remain.
    alg._tracking_env_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    step_count = torch.zeros(10, dtype=torch.long)
    for _ in range(20):
        alg._resample_replay_tracking(step_count, FakeReplay())
        idx = alg._replay_tracking_env_idx
        assert idx.numel() == 3
        assert set(idx.tolist()) == {7, 8, 9}


def test_replay_tracking_assigns_stepping_z():
    alg = _rollout_alg()
    replay = FakeReplay()
    step_count = torch.zeros(10, dtype=torch.long)
    z, _ = alg.maybe_update_rollout_context(None, step_count, expert_buffer=None, replay_buffer=replay)
    idx = alg._replay_tracking_env_idx
    assert idx.numel() == 3 and alg.replay_tracking_count == 3
    # z schedule: next-state rows 1..L, T=2 rolling mean -> at phase 0: mean(1,2)=1.5 (+base)
    zr = alg._replay_tracking_z
    assert zr.shape == (3, 5, 2)
    assert torch.allclose(zr[:, 0, 0] % 1000, torch.tensor([1.5, 1.5, 1.5]))
    assert torch.allclose(zr[:, 4, 0] % 1000, torch.tensor([5.0, 5.0, 5.0]))  # clamped at the end
    assert torch.allclose(z[idx], zr[:, 0])
    # non-tracking envs keep the sampled z
    others = torch.tensor([i for i in range(10) if i not in idx.tolist()])
    assert bool((z[others] == 99.0).all())
    # stepping: phase 1..4 follow the schedule
    for phase in range(1, 5):
        step_count += 1
        z, _ = alg.maybe_update_rollout_context(z, step_count, expert_buffer=None, replay_buffer=replay)
        assert torch.allclose(z[idx], zr[:, phase]), phase
    assert replay.calls == 1
    # wrap: new windows drawn, old envs released to the fresh z
    step_count += 1
    z, _ = alg.maybe_update_rollout_context(z, step_count, expert_buffer=None, replay_buffer=replay)
    assert replay.calls == 2 and alg._replay_tracking_phase == 0
    new_idx = alg._replay_tracking_env_idx
    released = torch.tensor([i for i in idx.tolist() if i not in new_idx.tolist()])
    if released.numel():
        assert bool((z[released] == 99.0).all())
    assert torch.allclose(z[new_idx], alg._replay_tracking_z[:, 0])


def test_replay_tracking_falls_back_when_replay_too_short():
    alg = _rollout_alg()
    replay = FakeReplay(ready=False)
    step_count = torch.zeros(10, dtype=torch.long)
    z, _ = alg.maybe_update_rollout_context(None, step_count, expert_buffer=None, replay_buffer=replay)
    assert alg.replay_tracking_count == 0 and bool((z == 99.0).all())
    # becomes ready later: picked up at the next window boundary
    replay.ready = True
    for _ in range(5):
        step_count += 1
        z, _ = alg.maybe_update_rollout_context(z, step_count, expert_buffer=None, replay_buffer=replay)
    assert alg.replay_tracking_count == 3


def test_replay_tracking_disabled_without_buffer_or_fraction():
    alg = _rollout_alg(frac=0.0)
    step_count = torch.zeros(10, dtype=torch.long)
    z, _ = alg.maybe_update_rollout_context(None, step_count, expert_buffer=None, replay_buffer=FakeReplay())
    assert alg.replay_tracking_count == 0 and bool((z == 99.0).all())
    alg = _rollout_alg(frac=0.3)
    z, _ = alg.maybe_update_rollout_context(None, step_count, expert_buffer=None, replay_buffer=None)
    assert alg.replay_tracking_count == 0
