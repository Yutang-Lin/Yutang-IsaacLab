# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Torch tests for BFM-0.7 replay "motion" support.

Covers the rolling-window mean helper, contiguous replay trajectory-window
sampling, and the expert buffer's zero placeholders for dataset-less obs keys.
"""

from __future__ import annotations

import importlib.util
import os

import torch

_HERE = os.path.dirname(__file__)


def _load(name: str, rel: str):
    path = os.path.join(_HERE, "..", "isaaclab_rl", "rsl_rl", *rel.split("/"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_storage = _load("fb_cpr_storage", "storage/fb_cpr_storage.py")
_math = _load("fb_cpr_math", "fb_cpr_math.py")
FBCprReplayBuffer = _storage.FBCprReplayBuffer
FBCprExpertBuffer = _storage.FBCprExpertBuffer


def test_rolling_window_mean_matches_naive_loop():
    torch.manual_seed(0)
    N, L, d = 5, 12, 3
    frames = torch.randn(N, L, d)
    window = torch.tensor([1, 2, 5, 12, 40])
    out = _math.rolling_window_mean(frames, window)
    for n in range(N):
        for t in range(L):
            ref = frames[n, t: min(t + int(window[n]), L)].mean(dim=0)
            assert torch.allclose(out[n, t], ref, atol=1e-6), (n, t)


def _make_buffer(time_cap=32, num_envs=1):
    return FBCprReplayBuffer(
        capacity=time_cap * num_envs,
        num_envs=num_envs,
        obs_space={"state": (1,), "other": (1,)},
        action_dim=1,
        z_dim=1,
        aux_reward_names=[],
        device="cpu",
        pin_memory=False,
    )


def _extend(buf, value, truncated):
    n = buf.num_envs
    buf.extend({
        "observation": {
            "state": torch.full((n, 1), float(value)),
            "other": torch.full((n, 1), -float(value)),
        },
        "action": torch.zeros(n, 1),
        "z": torch.zeros(n, 1),
        "terminated": torch.zeros(n, 1, dtype=torch.bool),
        "truncated": torch.full((n, 1), bool(truncated)),
    })


def test_sample_trajectory_windows_stay_in_segment_and_are_contiguous():
    buf = _make_buffer(time_cap=64, num_envs=2)
    # env 0 and env 1 share rows; segments: [0..9], [10..29], [30..39]
    for t in range(40):
        _extend(buf, t, truncated=(t in (0, 10, 30)))
    torch.manual_seed(0)
    L = 8
    out = buf.sample_trajectory_windows(64, L, keys=("state",))
    st = out["observation"]["state"]  # [64, L+1, 1]
    assert st.shape == (64, L + 1, 1)
    assert set(out["observation"].keys()) == {"state"}
    # contiguous increasing values
    diffs = st[:, 1:, 0] - st[:, :-1, 0]
    assert torch.equal(diffs, torch.ones_like(diffs))
    # never crosses a boundary: no window contains both 9 and 10, or 29 and 30
    first = st[:, 0, 0]
    last = st[:, -1, 0]
    assert not bool(((first <= 9) & (last >= 10)).any())
    assert not bool(((first <= 29) & (last >= 30)).any())
    # the 10-row segments can host a window of 9 rows (start 0 or 1 only)
    assert bool(((first >= 0) & (first <= 1) | (first >= 10)).all())


def test_sample_trajectory_windows_across_ring_wrap():
    buf = _make_buffer(time_cap=8)
    for t in range(11):  # ring holds values 3..10; newest at ring row 2
        _extend(buf, t, truncated=(t == 0))
    torch.manual_seed(0)
    out = buf.sample_trajectory_windows(32, 4, keys=("state",))
    st = out["observation"]["state"][:, :, 0]
    diffs = st[:, 1:] - st[:, :-1]
    assert torch.equal(diffs, torch.ones_like(diffs))
    assert bool((st.min(dim=1).values >= 3).all()) and bool((st.max(dim=1).values <= 10).all())


def test_sample_trajectory_windows_returns_none_when_too_short():
    buf = _make_buffer(time_cap=32)
    for t in range(6):
        _extend(buf, t, truncated=(t == 0))
    assert buf.sample_trajectory_windows(4, 8, keys=("state",)) is None
    assert buf.sample_trajectory_windows(4, 5, keys=("state",)) is not None
    assert _make_buffer().sample_trajectory_windows(4, 5, keys=("state",)) is None


def test_expert_placeholder_obs_adds_zero_keys_only_when_configured():
    buffer = FBCprExpertBuffer.__new__(FBCprExpertBuffer)
    obs = {"state": torch.ones(3, 2)}
    buffer._add_placeholder_obs(obs, 3, "cpu")   # no attribute -> no-op
    assert set(obs) == {"state"}
    buffer._placeholder_obs_dims = {"contact_labels": 4}
    buffer._add_placeholder_obs(obs, 3, "cpu")
    assert obs["contact_labels"].shape == (3, 4) and not obs["contact_labels"].any()
    # existing keys are never overwritten
    obs["contact_labels"] += 1
    buffer._add_placeholder_obs(obs, 3, "cpu")
    assert bool(obs["contact_labels"].all())
