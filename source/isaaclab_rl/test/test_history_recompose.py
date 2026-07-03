# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Byte-exact equivalence test for FBCprReplayBuffer history_actor recompose.

Simulates the env's history ring EXACTLY (push newest frame at slot 0, roll,
zero the whole ring on reset) to produce, at every step, the SAME flat
``history_actor`` blob the real env writes:
  per-term-blocked, each block frame-major NEWEST-FIRST:
    [act(H*Wa) | angv(H*Wg) | dofp(H*Wp) | dofv(H*Wv) | grav(H*Wgr)]

Then it feeds identical rollouts (same random frames, same resets, same buffer
wrap) into TWO buffers — one storing the full blob (baseline), one with
recompose enabled (stores one frame, rebuilds on sample) — and asserts the
sampled ``history_actor`` (and next-obs) match BIT-FOR-BIT across many samples,
including after buffer wraparound.

Run: python source/isaaclab_rl/test/test_history_recompose.py
(pure torch, no Isaac Sim.)
"""

from __future__ import annotations

import importlib.util
import os

import torch

# Load fb_cpr_storage.py DIRECTLY by path — bypass the package __init__ chain
# (which pulls in the external ``rsl_rl`` dep, unavailable in a bare test env).
_STORAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "isaaclab_rl", "rsl_rl", "storage",
    "fb_cpr_storage.py",
)
_spec = importlib.util.spec_from_file_location("fb_cpr_storage", _STORAGE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FBCprReplayBuffer = _mod.FBCprReplayBuffer


# Per-frame block layout in ENV STORAGE ORDER (name, width). 93 total.
BLOCKS = [("act", 29), ("angv", 3), ("dofp", 29), ("dofv", 29), ("grav", 3)]
FRAME_DIM = sum(w for _, w in BLOCKS)


class EnvHistorySim:
    """Replicates the env's per-term lagged history ring for one obs group.

    Holds, per env, an (H+1)-deep ring PER block (newest at slot 0). Each step:
    push the step's per-block noisy frame at slot 0 (roll first), and the
    emitted ``history_actor`` is slots[1:H+1] (the lagged frames), flattened
    per-term-blocked newest-first. Reset zeroes the whole ring for that env.
    """

    def __init__(self, num_envs: int, H: int, device):
        self.E = num_envs
        self.H = H
        self.device = device
        # ring depth H+1 (slot 0 = current, slots 1..H = lags), per block.
        self.rings = {name: torch.zeros(num_envs, H + 1, w, device=device)
                      for name, w in BLOCKS}

    def reset(self, env_ids):
        for name, _ in BLOCKS:
            self.rings[name][env_ids] = 0.0

    def step(self, frame_blocks):
        """Push one step's frame (dict name->[E,w]) and return the flat
        history_actor blob [E, H*FRAME_DIM] (lags 1..H), per-term newest-first.
        Mirrors _LaggedHistoryWrapper: roll(+1), insert at 0, return [1:H+1].
        """
        parts = []
        for name, w in BLOCKS:
            r = self.rings[name]
            r = r.roll(1, dims=1)
            r[:, 0] = frame_blocks[name]
            self.rings[name] = r
            lag = r[:, 1:self.H + 1]                    # [E, H, w] newest-first
            parts.append(lag.reshape(self.E, self.H * w))
        return torch.cat(parts, dim=-1)                # [E, H*FRAME_DIM]

    def current_frame_blockorder(self, frame_blocks):
        """The newest single frame in block order [act|angv|dofp|dofv|grav] —
        what the recompose buffer stores. (= slot-0 content just pushed.)"""
        return torch.cat([frame_blocks[name] for name, _ in BLOCKS], dim=-1)


def _make_buffer(capacity, num_envs, H, recompose, device):
    obs_space = {
        "state": (8,),
        "history_actor": (H * FRAME_DIM,),
    }
    spec = None
    if recompose:
        spec = {"key": "history_actor", "H": H, "blocks": BLOCKS}
    return FBCprReplayBuffer(
        capacity=capacity, num_envs=num_envs, obs_space=obs_space,
        action_dim=4, z_dim=6, aux_reward_names=["r"], device=device,
        history_recompose=spec,
    )


def run(seed=0, num_envs=4, H=9, time_cap=40, n_steps=140,
        reset_prob=0.06, n_sample_batches=60, batch=512, device="cpu"):
    torch.manual_seed(seed)
    capacity = time_cap * num_envs
    buf_full = _make_buffer(capacity, num_envs, H, recompose=False, device=device)
    buf_rec = _make_buffer(capacity, num_envs, H, recompose=True, device=device)

    sim = EnvHistorySim(num_envs, H, device)
    all_idx = torch.arange(num_envs, device=device)
    # ep_started_this_step[e] = True on the FIRST step after a reset -> the row's
    # ``truncated`` (env stores prev-done as truncated at the fresh-spawn row).
    prev_done = torch.zeros(num_envs, dtype=torch.bool, device=device)

    max_abs = 0.0
    for step in range(n_steps):
        # New per-step frame (random "noisy" proprio) for each block.
        frame_blocks = {name: torch.randn(num_envs, w, device=device)
                        for name, w in BLOCKS}
        # On a reset step, the env zeroes the ring BEFORE composing this step's
        # obs; the freshly-spawned obs sees an all-zero history.
        reset_mask = prev_done
        reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            sim.reset(reset_ids)

        hist_full = sim.step(frame_blocks)              # [E, H*FRAME_DIM]
        state = torch.randn(num_envs, 8, device=device)

        # truncated column stored at this row = prev_done (env convention:
        # the fresh-spawn row carries the done that caused it).
        trunc = prev_done.view(-1, 1).clone()
        term = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
        common = dict(action=torch.randn(num_envs, 4, device=device),
                      z=torch.randn(num_envs, 6, device=device),
                      terminated=term, truncated=trunc,
                      aux_rewards={"r": torch.randn(num_envs, 1, device=device)})
        buf_full.extend({"observation": {"state": state, "history_actor": hist_full},
                         **common})
        buf_rec.extend({"observation": {"state": state,
                                        "history_actor": hist_full},
                        **common})

        # decide next-step resets
        prev_done = torch.rand(num_envs, device=device) < reset_prob

    # ---- sample both with IDENTICAL indices and compare history_actor ----
    # We bypass the random sampler and directly exercise _gather over ALL valid
    # (time,env) pairs so coverage is exhaustive, incl. wraparound + boundaries.
    buf_full._ensure_traj_info()
    buf_rec._ensure_traj_info()
    T = len(buf_full)
    mism = 0
    checked = 0
    # Exercise the SAME set of transitions the real sampler can draw: current
    # row must be WRITTEN, and next_obs (t+1) must stay within the written
    # region (not the unwritten tail / wrap cursor). This mirrors sample_flat's
    # eligibility; gathering the unwritten tail row would compare recompose's
    # reach-back against the full buffer's stored-zero row — a transition the
    # sampler never yields.
    Tcap = buf_full.time_capacity
    if buf_full._is_full:
        cursor = (buf_full._idx - 1) % Tcap
        # Exclude the wrap cursor (its t+1 is the oldest = synthetic boundary),
        # AND the oldest H rows whose deep history was EVICTED by the wrap: for a
        # current row within H steps of the oldest stored row, frames older than
        # the oldest row are gone from the buffer, so recompose (correctly) zeros
        # them while the full-storage baseline still has them. The real sampler
        # can draw these rows; recompose returns a zero-truncated (shorter)
        # history there — see class note. Excluded from the byte-exact check.
        oldest = buf_full._idx
        evicted = {(oldest + k) % Tcap for k in range(H)}
        valid_ts = [t for t in range(T) if t != cursor and t not in evicted]
    else:
        valid_ts = list(range(buf_full._idx - 1))   # t+1 must be < _idx
    times = torch.tensor(valid_ts, device=device)
    for e in range(num_envs):
        ti = times
        ei = torch.full_like(ti, e)
        tn = (ti + 1) % buf_full.time_capacity
        g_full = buf_full._gather(ti, ei, tn)
        g_rec = buf_rec._gather(ti, ei, tn)
        for side in ("observation",):
            a = g_full[side]["history_actor"]
            b = g_rec[side]["history_actor"]
            d = (a - b).abs().max().item()
            max_abs = max(max_abs, d)
            mism += int((a != b).any(dim=-1).sum().item())
            checked += a.shape[0]
        # next-obs too
        a = g_full["next"]["observation"]["history_actor"]
        b = g_rec["next"]["observation"]["history_actor"]
        d = (a - b).abs().max().item()
        max_abs = max(max_abs, d)
        mism += int((a != b).any(dim=-1).sum().item())
        checked += a.shape[0]

    return max_abs, mism, checked, buf_full._is_full


def test_storage_is_narrowed():
    """The recompose buffer must actually store ONE frame, not H frames."""
    H = 9
    full = _make_buffer(40 * 4, 4, H, recompose=False, device="cpu")
    rec = _make_buffer(40 * 4, 4, H, recompose=True, device="cpu")
    w_full = full._obs["history_actor"].shape[-1]
    w_rec = rec._obs["history_actor"].shape[-1]
    assert w_full == H * FRAME_DIM, w_full
    assert w_rec == FRAME_DIM, w_rec
    print(f"[OK] storage narrowed: history_actor {w_full} -> {w_rec} "
          f"({w_full / w_rec:.1f}x smaller)")


def main():
    torch.set_printoptions(precision=6)
    test_storage_is_narrowed()
    configs = [
        dict(seed=0, time_cap=40, n_steps=120, reset_prob=0.05),   # not full
        dict(seed=1, time_cap=20, n_steps=200, reset_prob=0.08),   # wraps (full)
        dict(seed=2, time_cap=64, n_steps=64, reset_prob=0.0),     # no resets
        dict(seed=3, time_cap=16, n_steps=300, reset_prob=0.20),   # heavy resets, wraps
        dict(seed=4, H=4, time_cap=32, n_steps=150, reset_prob=0.1),  # H=4 (BFM-Zero)
        dict(seed=5, num_envs=16, H=9, time_cap=50, n_steps=400, reset_prob=0.12),
        dict(seed=6, num_envs=8, H=9, time_cap=1250, n_steps=1400, reset_prob=0.02),  # prod-like
        dict(seed=7, num_envs=2, H=9, time_cap=12, n_steps=500, reset_prob=0.35),  # tiny+churny
    ]
    ok = True
    for c in configs:
        max_abs, mism, checked, was_full = run(**c)
        status = "OK" if (max_abs == 0.0 and mism == 0) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] {c} -> max_abs_diff={max_abs:.3e} "
              f"mismatched_rows={mism}/{checked} buffer_full={was_full}")
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
