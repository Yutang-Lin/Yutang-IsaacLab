# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay / expert / z buffers for the BFM-Zero FB-CPR-Aux agent (rsl_rl port).

This module re-implements BFM-Zero's ``DictBuffer`` (per-transition replay),
``TrajectoryDictBuffer`` (expert trajectory slicer) and ``ZBuffer`` (rolling
z reservoir) with the rsl_rl conventions used by this codebase. No tensordict
or torch._pytree dependencies -- everything is plain dicts of tensors.

All three classes expose a common dict-of-tensors sample format:

    {
        "observation":    {state, privileged_state, last_action, history_actor},
        "action":         [B, action_dim],
        "z":              [B, z_dim],
        "next": {
            "observation":   {state, privileged_state, last_action, history_actor},
            "terminated":    [B, 1]  (bool),
        },
        "aux_rewards": {name: [B, 1]}    # train replay only
    }

Samples stay on the buffer's own device; the caller is responsible for moving
the batch to the training device.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _space_shape(space) -> tuple[int, ...]:
    """Return the per-sample shape of a ``gymnasium.spaces.Box``-like object."""
    return tuple(int(s) for s in space.shape)


def _index_obs_dict(obs: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in obs.items()}


# ---------------------------------------------------------------------------
# 1. FBCprReplayBuffer -- circular dict-of-tensors replay
# ---------------------------------------------------------------------------

class FBCprReplayBuffer:
    """Trajectory-aware 2D circular replay buffer matching BFM-Zero's
    ``TrajectoryDictBufferMultiDim``.

    Storage layout: ``[time_steps, num_envs, ...]`` — each ``extend()`` call
    appends one time-slice ``[1, num_envs, ...]`` across all parallel envs.
    Episode boundaries are marked by the ``truncated`` column; ``sample()``
    draws contiguous sub-sequences of length ``seq_length`` that never cross
    episode boundaries, exactly as BFM-Zero does.

    ``capacity`` is the total number of transitions (= ``time_steps * num_envs``).
    The time-axis length is ``capacity // num_envs``.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_space: Any,
        action_dim: int,
        z_dim: int,
        aux_reward_names: list[str],
        device: str | torch.device = "cpu",
        pin_memory: bool | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.time_capacity = int(capacity) // self.num_envs
        self.capacity = self.time_capacity  # __len__ counts time-steps
        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.z_dim = int(z_dim)
        # BFM's train replay uses seq_length=1 (individual transitions).
        # seq_length=8 is only for the expert slicer. The train buffer's
        # trajectory awareness is for episode-boundary safety, not for
        # returning contiguous multi-frame windows.
        self.seq_length = 1
        self.aux_reward_names = list(aux_reward_names)

        if pin_memory is None:
            pin_memory = self.device.type == "cpu"
        self._pin_memory = bool(pin_memory)

        if hasattr(obs_space, "spaces"):
            self._obs_shapes = {k: _space_shape(v) for k, v in obs_space.spaces.items()}
        else:
            self._obs_shapes = {k: tuple(v) for k, v in dict(obs_space).items()}

        # 2D storage: [time_capacity, num_envs, ...]
        self._obs: dict[str, torch.Tensor] = {
            k: self._alloc((self.time_capacity, self.num_envs, *shape))
            for k, shape in self._obs_shapes.items()
        }
        self._action = self._alloc((self.time_capacity, self.num_envs, self.action_dim))
        self._z = self._alloc((self.time_capacity, self.num_envs, self.z_dim))
        self._terminated = self._alloc((self.time_capacity, self.num_envs, 1), dtype=torch.bool)
        self._truncated = self._alloc((self.time_capacity, self.num_envs, 1), dtype=torch.bool)
        self._aux_rewards: dict[str, torch.Tensor] = {
            name: self._alloc((self.time_capacity, self.num_envs, 1)) for name in self.aux_reward_names
        }

        self._idx = 0
        self._is_full = False
        self._recompute_traj_info = True
        # Cached trajectory start/stop/length info (recomputed lazily).
        self._start_idx: torch.Tensor | None = None
        self._lengths: torch.Tensor | None = None

    def _alloc(self, shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self._pin_memory and self.device.type == "cpu":
            t = torch.empty(shape, dtype=dtype, pin_memory=True)
        else:
            t = torch.empty(shape, dtype=dtype, device=self.device)
        t.zero_()
        return t

    def __len__(self) -> int:
        return self.time_capacity if self._is_full else self._idx

    @property
    def full(self) -> bool:
        return self._is_full

    @property
    def total_transitions(self) -> int:
        return len(self) * self.num_envs

    # -- serialization ----------------------------------------------------

    @torch.no_grad()
    def state_dict(self) -> dict:
        return {
            "_obs": {k: v.clone() for k, v in self._obs.items()},
            "_action": self._action.clone(),
            "_z": self._z.clone(),
            "_terminated": self._terminated.clone(),
            "_truncated": self._truncated.clone(),
            "_aux_rewards": {k: v.clone() for k, v in self._aux_rewards.items()},
            "_idx": int(self._idx),
            "_is_full": bool(self._is_full),
            "time_capacity": self.time_capacity,
            "num_envs": self.num_envs,
            "action_dim": self.action_dim,
            "z_dim": self.z_dim,
            "aux_reward_names": list(self.aux_reward_names),
        }

    @torch.no_grad()
    def load_state_dict(self, sd: dict) -> None:
        if int(sd.get("time_capacity", sd.get("capacity", -1))) != self.time_capacity:
            raise ValueError("replay time_capacity mismatch")
        if int(sd.get("num_envs", -1)) != self.num_envs:
            raise ValueError("replay num_envs mismatch")
        for k in self._obs_shapes:
            if k in sd["_obs"]:
                self._obs[k].copy_(sd["_obs"][k].to(self._obs[k].device))
        self._action.copy_(sd["_action"].to(self._action.device))
        self._z.copy_(sd["_z"].to(self._z.device))
        self._terminated.copy_(sd["_terminated"].to(self._terminated.device))
        self._truncated.copy_(sd["_truncated"].to(self._truncated.device))
        for name in self.aux_reward_names:
            if name in sd["_aux_rewards"]:
                self._aux_rewards[name].copy_(
                    sd["_aux_rewards"][name].to(self._aux_rewards[name].device),
                )
        self._idx = int(sd["_idx"])
        self._is_full = bool(sd["_is_full"])
        self._recompute_traj_info = True

    # -- extend (one time-step across all envs) ----------------------------

    @torch.no_grad()
    def extend(self, batch_dict: dict) -> None:
        """Append one time-step slice ``[1, num_envs, ...]``.

        ``batch_dict`` keys: ``observation`` (dict), ``action`` [num_envs, A],
        ``z`` [num_envs, Z], ``terminated`` [num_envs, 1], ``truncated``
        [num_envs, 1], and optionally ``aux_rewards`` sub-dict.
        """
        obs = batch_dict["observation"]
        t = self._idx

        def _copy(dst: torch.Tensor, src: torch.Tensor) -> None:
            if src.device == dst.device:
                dst.copy_(src)
            else:
                dst.copy_(src, non_blocking=False)

        for k in self._obs_shapes:
            _copy(self._obs[k][t], obs[k])
        _copy(self._action[t], batch_dict["action"])
        _copy(self._z[t], batch_dict["z"])
        term = batch_dict["terminated"]
        if term.dtype != torch.bool:
            term = term.bool()
        if term.dim() == 1:
            term = term.unsqueeze(-1)
        _copy(self._terminated[t], term)
        trunc = batch_dict["truncated"]
        if trunc.dtype != torch.bool:
            trunc = trunc.bool()
        if trunc.dim() == 1:
            trunc = trunc.unsqueeze(-1)
        _copy(self._truncated[t], trunc)
        aux = batch_dict.get("aux_rewards", {})
        for name in self.aux_reward_names:
            if name not in aux:
                continue
            v = aux[name]
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            _copy(self._aux_rewards[name][t], v)

        self._idx = t + 1
        if self._idx >= self.time_capacity:
            self._is_full = True
            self._idx = 0
        self._recompute_traj_info = True

    # -- trajectory segmentation (BFM's find_start_stop_traj) ---------------

    def _ensure_traj_info(self) -> None:
        if not self._recompute_traj_info:
            return
        done = self._truncated[:len(self)].squeeze(-1)  # [T, E] bool
        T = done.shape[0]
        E = done.shape[1]
        starts_list: list[torch.Tensor] = []
        lengths_list: list[int] = []
        if self._is_full:
            cursor = (self._idx - 1) % self.time_capacity
            done_copy = done.clone()
            done_copy[cursor] = True
        else:
            done_copy = done.clone()
            done_copy[T - 1] = True
        for e in range(E):
            col = done_copy[:, e]
            ends = col.nonzero(as_tuple=False).squeeze(-1)
            if ends.numel() == 0:
                starts_list.append(torch.tensor([0, e], device=self.device))
                lengths_list.append(T)
                continue
            prev_end = -1
            for end_t in ends.tolist():
                start_t = (prev_end + 1) % T if prev_end >= 0 else 0
                if self._is_full and prev_end == -1:
                    start_t = (ends[-1].item() + 1) % T
                length = end_t - start_t + 1
                if length <= 0:
                    length += T
                starts_list.append(torch.tensor([start_t, e], device=self.device))
                lengths_list.append(length)
                prev_end = end_t
        self._start_idx = torch.stack(starts_list)  # [N_traj, 2]
        self._lengths = torch.tensor(lengths_list, device=self.device, dtype=torch.long)
        self._recompute_traj_info = False

    # -- sampling (BFM's get_idxs + _tensor_slices_from_startend) -----------

    @torch.no_grad()
    def sample(self, batch_size: int, seq_length: int | None = None) -> dict:
        seq_length = seq_length or self.seq_length
        if len(self) == 0:
            raise RuntimeError("FBCprReplayBuffer.sample() called on empty buffer")
        # Round batch down to a multiple of seq_length.
        self._ensure_traj_info()
        num_slices = max(1, batch_size // seq_length)
        # Episode "length" includes the truncated row (post-reset obs).
        # The last valid transition's next_obs is at position length-2 (0-indexed).
        # A window of seq_length obs frames needs next_obs at start+seq_length,
        # so we need start + seq_length <= length - 2, i.e. length >= seq_length + 2.
        min_len = seq_length + 2
        eligible = self._lengths >= min_len
        if not bool(eligible.any().item()):
            raise RuntimeError(
                f"No trajectories with length >= {min_len}; buffer too small or all episodes shorter."
            )
        eligible_idx = eligible.nonzero(as_tuple=False).squeeze(-1)
        eligible_lengths = self._lengths[eligible_idx]
        eligible_starts = self._start_idx[eligible_idx]

        traj_sel = torch.randint(eligible_idx.shape[0], (num_slices,), device=self.device)
        sel_lengths = eligible_lengths[traj_sel]
        sel_starts = eligible_starts[traj_sel]  # [num_slices, 2]

        # max start = length - seq_length - 2 so that last next_obs
        # at start + seq_length stays before the truncated row.
        end_point = (sel_lengths - seq_length - 2).clamp_min(0).to(torch.float32)
        relative_starts = (torch.rand(num_slices, device=self.device) * (end_point + 1.0)).floor().to(torch.long)

        time_starts = (sel_starts[:, 0] + relative_starts)  # [num_slices]
        env_ids = sel_starts[:, 1]  # [num_slices]

        arange = torch.arange(seq_length, device=self.device)  # [seq_length]
        # time indices: [num_slices, seq_length] -> flatten to [batch_size]
        time_idx = (time_starts.unsqueeze(1) + arange.unsqueeze(0)) % self.time_capacity
        time_idx = time_idx.reshape(-1)
        env_idx = env_ids.unsqueeze(1).expand(-1, seq_length).reshape(-1)
        # next-step indices (t+1)
        time_idx_next = (time_idx + 1) % self.time_capacity

        return self._gather(time_idx, env_idx, time_idx_next)

    def _gather(self, time_idx: torch.Tensor, env_idx: torch.Tensor,
                time_idx_next: torch.Tensor) -> dict:
        obs = {k: v[time_idx, env_idx] for k, v in self._obs.items()}
        next_obs = {k: v[time_idx_next, env_idx] for k, v in self._obs.items()}
        return {
            "observation": obs,
            "action": self._action[time_idx, env_idx],
            "z": self._z[time_idx, env_idx],
            "next": {
                "observation": next_obs,
                "terminated": self._terminated[time_idx_next, env_idx],
            },
            "aux_rewards": {
                name: self._aux_rewards[name][time_idx, env_idx]
                for name in self.aux_reward_names
            },
        }

    def sample_chunks(self, batch_size: int, num_chunks: int, target_device: str | torch.device) -> list[dict]:
        """Sample ``num_chunks`` batches in ONE call, then transfer to ``target_device``."""
        # Round per-chunk batch down to multiple of seq_length BEFORE
        # multiplying so all chunks are exactly batch_size.
        batch_size = max(self.seq_length, (int(batch_size) // self.seq_length) * self.seq_length)
        total = batch_size * int(num_chunks)
        big = self.sample(total, seq_length=self.seq_length)

        def _move(x: torch.Tensor) -> torch.Tensor:
            return x.to(target_device, non_blocking=True)

        obs_flat = {k: _move(v) for k, v in big["observation"].items()}
        next_obs_flat = {k: _move(v) for k, v in big["next"]["observation"].items()}
        action_flat = _move(big["action"])
        z_flat = _move(big["z"])
        term_flat = _move(big["next"]["terminated"])
        aux_flat = {name: _move(big["aux_rewards"][name]) for name in self.aux_reward_names}

        chunks: list[dict] = []
        for i in range(num_chunks):
            s = slice(i * batch_size, (i + 1) * batch_size)
            chunks.append({
                "observation": {k: obs_flat[k][s] for k in obs_flat},
                "action": action_flat[s],
                "z": z_flat[s],
                "next": {
                    "observation": {k: next_obs_flat[k][s] for k in next_obs_flat},
                    "terminated": term_flat[s],
                },
                "aux_rewards": {name: aux_flat[name][s] for name in self.aux_reward_names},
            })
        return chunks


# ---------------------------------------------------------------------------
# 2. FBCprExpertBuffer -- expert trajectory slicer loaded from .pt
# ---------------------------------------------------------------------------

class FBCprExpertBuffer:
    """Expert slicer that samples (s, s_next) windows of length ``seq_length``.

    Loads a ``.pt`` produced by ``scripts/precompute_bfm_expert_dataset.py``.
    Expected top-level keys: ``motions`` (dict name -> per-motion tensor dict),
    ``state_dim``, ``privileged_state_dim``, ``history_actor_dim``, etc.

    Each per-motion dict is expected to carry ``state`` [T, D_s],
    ``privileged_state`` [T, D_p], ``last_action`` [T, 29],
    ``history_actor`` [T, D_h].

    Sampling strategy (matches BFM ``TrajectoryDictBuffer.sample``): pick
    ``batch_size // seq_length`` trajectory start indices (optionally weighted
    by per-motion priorities), expand each into a length-``seq_length`` window,
    then flatten to a flat batch of shape ``(batch_size,)`` rows. The next-step
    observation is the same window shifted by +1 (so each motion needs at
    least ``seq_length + 1`` frames to be eligible).
    """

    def __init__(
        self,
        pt_path: str,
        seq_length: int,
        device: str | torch.device = "cpu",
        motion_ids: list[str] | None = None,
        length_proportional_priors: bool = True,
        distributed_shard: bool = False,
        shard_rank: int = 0,
        shard_world_size: int = 1,
        shard_seed: int = 0,
    ) -> None:
        """Expert motion buffer.

        Args:
            length_proportional_priors: If True (default), initial
                ``_priorities`` are set proportional to each motion's
                frame count, so the expected per-transition draw
                probability is uniform across the dataset. With
                unbalanced clip lengths (e.g. a 7468-frame continuous
                motion alongside 8000-frame LAFAN clips) this matters a
                lot: under uniform motion-priors the transitions from a
                short clip are sampled ~clip-length-ratio more often per
                update. ``update_priorities()`` (from the tracking eval)
                overwrites this init.
        """
        self.seq_length = int(seq_length)
        self._length_proportional_priors = bool(length_proportional_priors)
        self.device = torch.device(device)

        raw = torch.load(pt_path, weights_only=False, map_location="cpu")
        if not isinstance(raw, dict) or "motions" not in raw:
            raise ValueError(
                f"Expected a dict with 'motions' key from {pt_path}, got {type(raw)}"
            )

        # ------------------------------------------------------------- #
        # Minimal-format dataset: motion dicts only carry raw fields
        # (root_pos, root_quat, joint_pos, fps, ...). We derive the full
        # obs/RSI set at load time by calling the precompute pipeline's
        # ``_process_motion`` on each motion (per this rank's shard when
        # distributed). Avoids shipping huge precomputed buffers on disk.
        # ------------------------------------------------------------- #
        self._minimal: bool = bool(raw.get("minimal", False))
        self._minimal_derive_fn = None
        if self._minimal:
            # Lazy-import the precompute pipeline. It lives in the
            # Latent-Control repo under scripts/; the runner adds it to
            # sys.path via the cfg's ``expert_dataset_compose_script``
            # (with a sane default relative to this file's location).
            import importlib.util, os, sys
            script_rel = os.environ.get(
                "BFM_EXPERT_COMPOSE_SCRIPT",
                # Default: Latent-Control/scripts/precompute_bfm_expert_dataset.py
                # one sibling up from Yutang-IsaacLab.
                os.path.abspath(os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "..", "..", "..", "..",
                    "Latent-Control", "scripts",
                    "precompute_bfm_expert_dataset.py",
                )),
            )
            if not os.path.exists(script_rel):
                raise FileNotFoundError(
                    f"Minimal expert dataset {pt_path} requires the precompute "
                    f"script at {script_rel} (not found). Set BFM_EXPERT_COMPOSE_SCRIPT "
                    f"env var to its absolute path, or rebuild the dataset "
                    f"without --minimal."
                )
            spec = importlib.util.spec_from_file_location(
                "_bfm_precompute_loader", script_rel,
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_bfm_precompute_loader"] = mod
            spec.loader.exec_module(mod)
            self._minimal_mod = mod
            urdf = raw.get("urdf_path") or mod.DEFAULT_URDF
            # Build the FK chain on the buffer's device so per-motion
            # compose runs entirely on GPU — ~20-40x faster than CPU on
            # a big dataset. When self.device is "cuda", pytorch_kinematics
            # moves its internal state to the same device.
            compose_device = str(self.device)
            self._minimal_chain = mod._build_chain(urdf, device=compose_device)
            self._minimal_default_q = torch.tensor(
                [float(x) for x in raw.get("default_dof_pos", mod.DEFAULT_DOF_POS)],
                dtype=torch.float32, device=self.device,
            )
            self._minimal_gravity = torch.tensor(
                list(raw.get("gravity", [0.0, 0.0, -1.0])),
                dtype=torch.float32, device=self.device,
            )
            self._minimal_history_len = int(raw.get("history_length", 4))
            self._minimal_action_scale = float(raw.get("action_scale", 0.25))
            self._minimal_action_clip = float(raw.get("action_clip", 5.0))
            self._minimal_resample_fps = float(raw.get("resample_fps", 0.0)) or None
            print(
                f"[FBCprExpertBuffer] minimal dataset detected; "
                f"load-time compose pipeline ready "
                f"(urdf={os.path.basename(urdf)}).",
                flush=True,
            )

        all_motions = raw["motions"]
        if motion_ids is None:
            motion_ids = list(all_motions.keys())
        else:
            missing = [m for m in motion_ids if m not in all_motions]
            if missing:
                raise KeyError(f"Motions not in dataset: {missing}")

        # Distributed shard: randomly permute the motion list with a seed
        # shared across ranks (so every rank gets a deterministic global
        # permutation), then take the contiguous slice belonging to this
        # rank. Result: each motion is owned by exactly ONE rank, and the
        # assignment is shuffled so locomotion/manipulation/stair/etc. are
        # spread across ranks rather than clumped by insertion order.
        self._shard_info: dict[str, int] = {
            "rank": int(shard_rank) if distributed_shard else 0,
            "world_size": int(shard_world_size) if distributed_shard else 1,
            "global_num_motions": len(motion_ids),
            "local_num_motions": len(motion_ids),
        }
        if distributed_shard and shard_world_size > 1:
            n_total = len(motion_ids)
            g = torch.Generator(device="cpu").manual_seed(int(shard_seed))
            perm = torch.randperm(n_total, generator=g).tolist()
            # Even split: rank r gets indices perm[r::W] for balanced load.
            local_idxs = [perm[i] for i in range(shard_rank, n_total, shard_world_size)]
            motion_ids = [motion_ids[i] for i in local_idxs]
            self._shard_info["local_num_motions"] = len(motion_ids)
            print(
                f"[FBCprExpertBuffer] distributed shard rank "
                f"{shard_rank}/{shard_world_size}: {len(motion_ids)}/{n_total} motions "
                f"(seed={shard_seed})",
                flush=True,
            )

        self._motion_names: list[str] = list(motion_ids)
        self._states: list[torch.Tensor] = []
        self._privs: list[torch.Tensor] = []
        self._last_actions: list[torch.Tensor] = []
        self._history_actors: list[torch.Tensor] = []
        self._lengths: list[int] = []

        # RSI reset-state fields, populated below if present in the .pt.
        joint_pos_chunks: list[torch.Tensor] = []
        joint_vel_chunks: list[torch.Tensor] = []
        root_pos_chunks: list[torch.Tensor] = []
        root_quat_chunks: list[torch.Tensor] = []
        root_lin_vel_chunks: list[torch.Tensor] = []
        root_ang_vel_chunks: list[torch.Tensor] = []
        reset_fields_complete = True

        # Per-motion metadata for BFM-Terrain (and any other task that needs
        # to route RSI by motion source). ``motion_source_id`` defaults to 0
        # (= "regular" / flat-floor) if the .pt doesn't carry the field;
        # ``requires_terrain`` defaults to False. ``terrain_mesh_path`` may
        # be used by the env to build its shared terrain mesh.
        self._motion_source_id: list[int] = []
        self._requires_terrain: list[bool] = []
        self._terrain_mesh_paths: list[str] = []

        # Minimal path: batch-FK across all shard motions once, then
        # loop per-motion over the fast post-FK pipeline. FK is strongly
        # Python-launch-bound per call, so concat along T and one kernel
        # dispatch is ~30x faster than calling FK per motion. Velocity
        # FD must stay per-motion (it can't cross clip boundaries), but
        # that step is cheap.
        batched_fk_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if self._minimal:
            mod = self._minimal_mod
            dev = self.device
            # Collect raw tensors + lengths.
            raw_rp: list[torch.Tensor] = []
            raw_rq: list[torch.Tensor] = []
            raw_jp: list[torch.Tensor] = []
            lens: list[int] = []
            ordered_names: list[str] = list(self._motion_names)
            for name in ordered_names:
                m = all_motions[name]
                rp = m["root_pos"].to(dev, non_blocking=True)
                rq = mod._quat_normalize(m["root_quat"].to(dev, non_blocking=True))
                jp = m["joint_pos"].to(dev, non_blocking=True)
                raw_rp.append(rp); raw_rq.append(rq); raw_jp.append(jp)
                lens.append(int(rp.shape[0]))
            cat_rp = torch.cat(raw_rp, dim=0)
            cat_rq = torch.cat(raw_rq, dim=0)
            cat_jp = torch.cat(raw_jp, dim=0)
            # One batched FK call.
            print(
                f"[FBCprExpertBuffer] batch FK on {len(ordered_names)} motions, "
                f"{int(cat_rp.shape[0])} total frames ...",
                flush=True,
            )
            cat_wp, cat_wq = mod._world_fk(
                self._minimal_chain, cat_jp, cat_rp, cat_rq, mod.KEYPOINT_NAMES,
            )
            # Split back into per-motion FK outputs.
            offs = 0
            for name, T in zip(ordered_names, lens):
                batched_fk_outputs[name] = (
                    cat_wp[offs:offs + T].contiguous(),
                    cat_wq[offs:offs + T].contiguous(),
                )
                offs += T
            # Cache raw pieces too so per-motion post-FK step avoids a
            # second copy.
            batched_raw = {
                name: (raw_rp[i], raw_rq[i], raw_jp[i])
                for i, name in enumerate(ordered_names)
            }
            # Free the concatenated scratch ASAP.
            del cat_rp, cat_rq, cat_jp, cat_wp, cat_wq, raw_rp, raw_rq, raw_jp

        # Progress bar for load-time compose (minimal datasets only).
        # Falls back to a silent no-op iterator when tqdm is unavailable
        # or the dataset is already precomputed.
        motion_iter = self._motion_names
        if self._minimal:
            try:
                from tqdm import tqdm
                shard_tag = ""
                if self._shard_info["world_size"] > 1:
                    shard_tag = (
                        f" [rank {self._shard_info['rank']}/"
                        f"{self._shard_info['world_size']}]"
                    )
                motion_iter = tqdm(
                    self._motion_names,
                    desc=f"[ExpertBuffer] compose{shard_tag}",
                    unit="motion",
                    dynamic_ncols=True,
                    mininterval=0.5,
                )
            except ImportError:
                pass

        for name in motion_iter:
            m = all_motions[name]
            # Minimal-dataset path: materialise state / priv / history /
            # velocities from the raw fields via _process_motion. We run
            # on CPU to keep torch JIT graph simple; the buffer moves to
            # GPU below. On a massive sharded dataset, this is the
            # dominant load-time cost (~5 ms per 500-frame motion).
            if self._minimal:
                mod = self._minimal_mod
                rp, rq, jp = batched_raw[name]
                wp_src, wq_src = batched_fk_outputs[name]
                src_fps_i = int(m["fps"])
                m = mod._process_motion_from_fk(
                    name=name,
                    root_pos=rp, root_quat=rq, joint_pos=jp,
                    world_pos_src=wp_src, world_quat_src=wq_src,
                    src_fps=src_fps_i, dt_min=1e-3,
                    keypoint_names=mod.KEYPOINT_NAMES,
                    default_q=self._minimal_default_q,
                    gravity_world=self._minimal_gravity,
                    history_length=self._minimal_history_len,
                    action_scale=self._minimal_action_scale,
                    action_clip=self._minimal_action_clip,
                    resample_fps=self._minimal_resample_fps,
                    terrain_mesh=None,
                    terrain_z_precomputed=(
                        all_motions[name]["terrain_z"]
                        if "terrain_z" in all_motions[name] else None
                    ),
                )
                if m is None:
                    raise RuntimeError(
                        f"Minimal compose returned None for motion {name!r} "
                        f"(clip likely too short, T<3). Drop it from the dataset."
                    )
                # Carry over the original per-motion metadata tags.
                m["motion_source_id"] = int(all_motions[name].get("motion_source_id", 0))
                m["requires_terrain"] = bool(all_motions[name].get("requires_terrain", False))
                m["terrain_id"] = int(all_motions[name].get("terrain_id", -1))
                m["terrain_mesh_path"] = str(all_motions[name].get("terrain_mesh_path", ""))
            self._states.append(m["state"].to(self.device).contiguous())
            self._privs.append(m["privileged_state"].to(self.device).contiguous())
            self._last_actions.append(m["last_action"].to(self.device).contiguous())
            self._history_actors.append(m["history_actor"].to(self.device).contiguous())
            self._lengths.append(int(m["state"].shape[0]))
            # RSI chunks — accumulate every motion or mark RSI unavailable.
            for key, bucket in (
                ("joint_pos", joint_pos_chunks),
                ("joint_vel", joint_vel_chunks),
                ("root_pos", root_pos_chunks),
                ("root_quat", root_quat_chunks),
                ("root_lin_vel", root_lin_vel_chunks),
                ("root_ang_vel", root_ang_vel_chunks),
            ):
                if key in m and isinstance(m[key], torch.Tensor):
                    bucket.append(m[key].to(self.device).contiguous())
                else:
                    reset_fields_complete = False
            # Per-motion tags (default-safe for legacy .pt files).
            self._motion_source_id.append(int(m.get("motion_source_id", 0)))
            self._requires_terrain.append(bool(m.get("requires_terrain", False)))
            self._terrain_mesh_paths.append(str(m.get("terrain_mesh_path", "")))

        self._lengths_t = torch.tensor(self._lengths, dtype=torch.long, device=self.device)

        # Per-motion RSI/tracking tensors, kept around so the tracking eval
        # can address individual motions (BFM's per-motion evaluation).
        self._per_motion_joint_pos: list[torch.Tensor] = list(joint_pos_chunks) if reset_fields_complete else []
        self._per_motion_joint_vel: list[torch.Tensor] = list(joint_vel_chunks) if reset_fields_complete else []
        self._per_motion_root_pos: list[torch.Tensor] = list(root_pos_chunks) if reset_fields_complete else []
        self._per_motion_root_quat: list[torch.Tensor] = list(root_quat_chunks) if reset_fields_complete else []
        self._per_motion_root_lin_vel: list[torch.Tensor] = list(root_lin_vel_chunks) if reset_fields_complete else []
        self._per_motion_root_ang_vel: list[torch.Tensor] = list(root_ang_vel_chunks) if reset_fields_complete else []

        # Infer dims from the first motion.
        self._state_dim = int(self._states[0].shape[-1])
        self._priv_dim = int(self._privs[0].shape[-1])
        self._action_dim = int(self._last_actions[0].shape[-1])
        self._history_actor_dim = int(self._history_actors[0].shape[-1])

        # RSI support: only enabled when every motion has all the needed fields.
        self.supports_reset_states = reset_fields_complete and len(joint_pos_chunks) == len(self._motion_names)
        if self.supports_reset_states:
            self.joint_pos_buffer = torch.cat(joint_pos_chunks, dim=0).contiguous()
            self.joint_vel_buffer = torch.cat(joint_vel_chunks, dim=0).contiguous()
            self.root_pos_buffer = torch.cat(root_pos_chunks, dim=0).contiguous()
            self.root_quat_buffer = torch.cat(root_quat_chunks, dim=0).contiguous()
            self.root_lin_vel_buffer = torch.cat(root_lin_vel_chunks, dim=0).contiguous()
            self.root_ang_vel_buffer = torch.cat(root_ang_vel_chunks, dim=0).contiguous()
            self.total_frames = int(self.joint_pos_buffer.shape[0])
            self.num_joints = int(self.joint_pos_buffer.shape[1])
            # Per-motion start offsets into the flat buffer (lazy RSI lookup).
            # ``_motion_starts[m]`` = first flat index, ``_motion_lengths_rsi[m]`` = frame count.
            _offs = [0]
            for chunk in joint_pos_chunks:
                _offs.append(_offs[-1] + int(chunk.shape[0]))
            self._motion_starts = torch.tensor(_offs[:-1], dtype=torch.long, device=self.device)
            self._motion_lengths_rsi = torch.tensor(
                [int(c.shape[0]) for c in joint_pos_chunks],
                dtype=torch.long, device=self.device,
            )
        else:
            self.joint_pos_buffer = None
            self.joint_vel_buffer = None
            self.root_pos_buffer = None
            self.root_quat_buffer = None
            self.root_lin_vel_buffer = None
            self.root_ang_vel_buffer = None
            self.total_frames = 0
            self.num_joints = 0

        # Initial priors: length-proportional (default) so the per-
        # transition draw probability is equal across motions — otherwise
        # a short clip's transitions are seen (clip_length_ratio)x more
        # often per update than a long clip's. With a mix of 8k-frame
        # LAFAN clips and a 7468-frame continuous motion that matters.
        # ``update_priorities()`` (from the tracking eval) overwrites this.
        if (
            self._length_proportional_priors
            and len(self._lengths) > 0
            and sum(int(x) for x in self._lengths) > 0
        ):
            self._priorities = torch.tensor(
                [float(x) for x in self._lengths],
                dtype=torch.float32, device=self.device,
            )
            self._priorities = self._priorities / self._priorities.sum().clamp_min(1e-12)
        else:
            self._priorities = torch.ones(
                len(self._motion_names), dtype=torch.float32, device=self.device,
            )
            self._priorities = self._priorities / self._priorities.sum().clamp_min(1e-12)

        # --- Flat concatenated obs buffers for O(1) sample() --------------
        # Same trick as the RSI flat buffer: cat all motions along time,
        # keep per-motion ``_motion_obs_starts`` offsets, then sample()
        # can do ``flat[global_frame_idx]`` with ONE indexed read per leaf
        # instead of a Python for-loop over unique motions. Cost: 2× RAM
        # on the expert dataset (~2 GB here), which is fine on GPU.
        self._flat_state = torch.cat(self._states, dim=0).contiguous()
        self._flat_priv = torch.cat(self._privs, dim=0).contiguous()
        self._flat_last_action = torch.cat(self._last_actions, dim=0).contiguous()
        self._flat_history_actor = torch.cat(self._history_actors, dim=0).contiguous()
        _obs_offs = [0]
        for m in self._states:
            _obs_offs.append(_obs_offs[-1] + int(m.shape[0]))
        self._motion_obs_starts = torch.tensor(
            _obs_offs[:-1], dtype=torch.long, device=self.device,
        )  # [num_motions]; absolute start in flat_*.

        # Per-motion metadata as tensors (for device-efficient gather).
        self.motion_source_id_t = torch.tensor(
            self._motion_source_id, dtype=torch.long, device=self.device,
        )
        self.requires_terrain_t = torch.tensor(
            self._requires_terrain, dtype=torch.bool, device=self.device,
        )
        # When RSI fields are available, expose per-RSI-frame metadata via
        # the flat buffer ordering. ``_motion_lengths_rsi`` gives the frame
        # count per motion; ``repeat_interleave`` maps that back to per-frame.
        if self.supports_reset_states:
            self.frame_motion_source_id = torch.repeat_interleave(
                self.motion_source_id_t, self._motion_lengths_rsi,
            )  # [total_frames]
            self.frame_requires_terrain = torch.repeat_interleave(
                self.requires_terrain_t, self._motion_lengths_rsi,
            )  # [total_frames]
        else:
            self.frame_motion_source_id = torch.zeros(0, dtype=torch.long, device=self.device)
            self.frame_requires_terrain = torch.zeros(0, dtype=torch.bool, device=self.device)

    # -- properties --------------------------------------------------------

    @property
    def motion_ids(self) -> list[str]:
        return list(self._motion_names)

    @property
    def file_names(self) -> list[str]:
        return list(self._motion_names)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def priv_dim(self) -> int:
        return self._priv_dim

    @property
    def history_actor_dim(self) -> int:
        return self._history_actor_dim

    def __len__(self) -> int:
        return len(self._motion_names)

    def empty(self) -> bool:
        return len(self._motion_names) == 0

    @property
    def num_unique_motions(self) -> int:
        return len(self._motion_names)

    @property
    def lengths(self) -> list[int]:
        return list(self._lengths)

    @torch.no_grad()
    def get_motion_window(self, motion_id: int, num_frames: int) -> dict:
        """Return the first ``num_frames`` frames of motion ``motion_id`` as a dict.

        Truncates if the motion is shorter than ``num_frames`` (returns its
        actual length), so callers should read ``result['num_frames']``
        rather than assuming the full request.
        """
        assert 0 <= motion_id < self.num_unique_motions
        L = min(int(num_frames), self._lengths[motion_id])
        out = {
            "state": self._states[motion_id][:L],
            "privileged_state": self._privs[motion_id][:L],
            "last_action": self._last_actions[motion_id][:L],
            "history_actor": self._history_actors[motion_id][:L],
            "num_frames": L,
        }
        if self.supports_reset_states:
            out.update({
                "joint_pos": self._per_motion_joint_pos[motion_id][:L],
                "joint_vel": self._per_motion_joint_vel[motion_id][:L],
                "root_pos": self._per_motion_root_pos[motion_id][:L],
                "root_quat": self._per_motion_root_quat[motion_id][:L],
                "root_lin_vel": self._per_motion_root_lin_vel[motion_id][:L],
                "root_ang_vel": self._per_motion_root_ang_vel[motion_id][:L],
            })
        return out

    @torch.no_grad()
    def compute_body_pos(self, global_frames: torch.Tensor) -> torch.Tensor | None:
        """Run FK for the given flat-buffer frame indices, return [N, K, 3]."""
        if not self.supports_reset_states:
            return None
        mod = getattr(self, "_minimal_mod", None)
        chain = getattr(self, "_minimal_chain", None)
        if mod is None or chain is None:
            return None
        idx = global_frames.to(self.joint_pos_buffer.device)
        jp = self.joint_pos_buffer[idx]
        rp = self.root_pos_buffer[idx]
        rq = self.root_quat_buffer[idx]
        wp, _ = mod._world_fk(chain, jp, rp, rq, mod.KEYPOINT_NAMES)
        return wp  # [N, K, 3]

    # -- RSI -------------------------------------------------------------- #

    @torch.no_grad()
    def get_reset_states_at(self, motion_ids: torch.Tensor, frame_offsets: torch.Tensor) -> dict:
        """Return RSI state for specific motion/frame pairs."""
        starts = self._motion_starts[motion_ids]
        lens = self._motion_lengths_rsi[motion_ids]
        offsets = frame_offsets.clamp(max=lens - 1)
        frame = starts + offsets
        return {
            "joint_pos": self.joint_pos_buffer[frame],
            "joint_vel": self.joint_vel_buffer[frame],
            "root_pos": self.root_pos_buffer[frame],
            "root_quat": self.root_quat_buffer[frame],
            "root_lin_vel": self.root_lin_vel_buffer[frame],
            "root_ang_vel": self.root_ang_vel_buffer[frame],
            "motion_source_id": self.frame_motion_source_id[frame],
            "requires_terrain": self.frame_requires_terrain[frame],
        }

    @torch.no_grad()
    def sample_reset_states(self, batch_size: int) -> dict:
        """Sample expert frames for reference-state initialization (BFM-style RSI).

        Uniform over every frame in the flat buffer; returns world-frame root
        pose + velocities and joint pose + velocities so the env can write
        them into the simulator directly.
        """
        if not self.supports_reset_states:
            raise RuntimeError(
                "FBCprExpertBuffer does not have RSI fields. Re-run "
                "scripts/precompute_bfm_expert_dataset.py so that "
                "joint_pos / joint_vel / root_pos / root_quat / root_lin_vel "
                "/ root_ang_vel are populated for every motion."
            )
        # BFM's RSI samples in two stages: (a) motion_id ~ sampling_weights,
        # (b) start_time ~ Uniform[0, motion_length). Flat-uniform frame
        # sampling (old code) over-represents long motions and bypasses
        # priority feedback entirely. This restores BFM's semantics.
        motion_ids = torch.multinomial(
            self._priorities, num_samples=batch_size, replacement=True,
        )
        starts = self._motion_starts[motion_ids]
        lens = self._motion_lengths_rsi[motion_ids]
        # Uniform frame within each motion.
        rand = torch.rand(batch_size, device=self.device)
        offsets = (rand * lens.to(torch.float32)).floor().to(torch.long).clamp(max=lens - 1)
        frame = starts + offsets
        return {
            "joint_pos": self.joint_pos_buffer[frame],
            "joint_vel": self.joint_vel_buffer[frame],
            "root_pos": self.root_pos_buffer[frame],
            "root_quat": self.root_quat_buffer[frame],
            "root_lin_vel": self.root_lin_vel_buffer[frame],
            "root_ang_vel": self.root_ang_vel_buffer[frame],
            "motion_source_id": self.frame_motion_source_id[frame],
            "requires_terrain": self.frame_requires_terrain[frame],
        }

    def sample_tracking_trajectories(
        self, num_trajs: int, traj_length: int,
    ) -> dict:
        """Sample contiguous expert sub-trajectories for tracking.

        Returns a dict with:
            - "observation": dict of obs tensors [B, ...]  (B = num_trajs * traj_length)
            - "next_observation": dict [B, ...]
            - "motion_ids": [num_trajs] — which motion each trajectory is from
            - "starts": [num_trajs] — start frame within each motion
            - "motion_lens": [num_trajs] — usable length of each motion
        """
        MIN_FRAMES = 50
        eligible_mask = self._lengths_t >= MIN_FRAMES
        if not bool(eligible_mask.any().item()):
            raise RuntimeError(f"No motion has at least {MIN_FRAMES} frames.")
        eligible_idx = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        eligible_priors = self._priorities[eligible_idx]
        eligible_priors = eligible_priors / eligible_priors.sum().clamp_min(1e-12)
        eligible_lengths = self._lengths_t[eligible_idx]

        sel = torch.multinomial(eligible_priors, num_trajs, replacement=True)
        motion_picks = eligible_idx[sel]
        motion_lens = eligible_lengths[sel]
        rand01 = torch.rand(num_trajs, device=self.device)
        max_start = (motion_lens - traj_length - 1).clamp_min(0).to(torch.float32)
        starts = (rand01 * (max_start + 1.0)).floor().to(torch.long)

        arange = torch.arange(traj_length, device=self.device).unsqueeze(0)
        raw_frame = starts.unsqueeze(1) + arange
        usable_lens = (motion_lens - 1).clamp_min(1).unsqueeze(1)
        # Terrain motions: clamp to last frame (pad). Others: circular wrap.
        is_terrain = self.requires_terrain_t[motion_picks].unsqueeze(1)  # [N, 1]
        frame_cur_wrap = raw_frame % usable_lens
        frame_cur_clamp = raw_frame.clamp(max=usable_lens - 1)
        frame_cur = torch.where(is_terrain, frame_cur_clamp, frame_cur_wrap).reshape(-1)
        frame_nxt_wrap = (raw_frame + 1) % usable_lens
        frame_nxt_clamp = (raw_frame + 1).clamp(max=usable_lens - 1)
        frame_nxt = torch.where(is_terrain, frame_nxt_clamp, frame_nxt_wrap).reshape(-1)
        motion_flat = motion_picks.unsqueeze(1).expand(-1, traj_length).reshape(-1)

        global_cur = self._motion_obs_starts[motion_flat] + frame_cur
        global_nxt = self._motion_obs_starts[motion_flat] + frame_nxt

        obs = {
            "state": self._flat_state[global_cur],
            "privileged_state": self._flat_priv[global_cur],
            "last_action": self._flat_last_action[global_cur],
            "history_actor": self._flat_history_actor[global_cur],
        }
        next_obs = {
            "state": self._flat_state[global_nxt],
            "privileged_state": self._flat_priv[global_nxt],
            "last_action": self._flat_last_action[global_nxt],
            "history_actor": self._flat_history_actor[global_nxt],
        }
        return {
            "observation": obs,
            "next_observation": next_obs,
            "motion_ids": motion_picks,
            "starts": starts,
            "motion_lens": motion_lens,
            "requires_terrain": self.requires_terrain_t[motion_picks],
        }

    # -- priority updates (stub-ish; accepted by agent) --------------------

    def update_priorities(self, priorities: torch.Tensor, idxs: torch.Tensor | None = None) -> None:
        """Update per-motion sampling weights.

        If ``idxs`` is None, expects ``priorities`` to have length equal to
        the number of motions. Otherwise scatters the new values into the
        given indices. Non-negative values are required; values are then
        renormalised to sum to 1.

        When ``length_proportional_priors`` is set, the incoming weights
        (e.g. MPJPE-derived) are multiplied by per-motion length BEFORE
        normalisation so the expected per-transition draw probability
        stays uniform across motions regardless of clip-length skew.
        """
        priorities = priorities.to(self.device).float().clamp_min(0.0)
        if idxs is None:
            if priorities.numel() != len(self._motion_names):
                raise ValueError(
                    f"Expected priorities of length {len(self._motion_names)}, got {priorities.numel()}"
                )
            if self._length_proportional_priors:
                lens = torch.tensor(
                    [float(x) for x in self._lengths],
                    dtype=priorities.dtype, device=self.device,
                )
                priorities = priorities * lens
            self._priorities = priorities
        else:
            idxs = idxs.to(self.device).long()
            if self._length_proportional_priors:
                lens = torch.tensor(
                    [float(self._lengths[int(i)]) for i in idxs.tolist()],
                    dtype=priorities.dtype, device=self.device,
                )
                priorities = priorities * lens
            self._priorities[idxs] = priorities
        s = self._priorities.sum()
        if s > 0:
            self._priorities = self._priorities / s
        else:
            self._priorities = torch.ones_like(self._priorities) / self._priorities.numel()

    # -- sampling ----------------------------------------------------------

    @torch.no_grad()
    def sample(self, batch_size: int, seq_length: int | None = None) -> dict:
        seq_length = int(seq_length) if seq_length is not None else self.seq_length
        # Round batch down to a multiple of seq_length.
        num_slices = max(1, batch_size // seq_length)

        # Eligibility: need at least 50 frames (~1s at 50Hz). Short motions
        # are circular-padded so they can be sampled for tracking trajectories
        # even if shorter than seq_length.
        MIN_FRAMES = 50
        eligible_mask = self._lengths_t >= MIN_FRAMES
        if not bool(eligible_mask.any().item()):
            raise RuntimeError(f"No motion has at least {MIN_FRAMES} frames.")
        eligible_idx = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        eligible_priors = self._priorities[eligible_idx]
        eligible_priors = eligible_priors / eligible_priors.sum().clamp_min(1e-12)
        eligible_lengths = self._lengths_t[eligible_idx]

        # Pick a motion per slice (weighted), then a random start.
        sel = torch.multinomial(eligible_priors, num_slices, replacement=True)
        motion_picks = eligible_idx[sel]                    # [num_slices]
        motion_lens = eligible_lengths[sel]                 # [num_slices]
        # For motions longer than seq_length: start in [0, T-seq_length-1].
        # For shorter motions: start at 0, frames will wrap via modulo.
        rand01 = torch.rand(num_slices, device=self.device)
        max_start = (motion_lens - seq_length - 1).clamp_min(0).to(torch.float32)
        starts = (rand01 * (max_start + 1.0)).floor().to(torch.long)

        # Build per-slice arange window and flatten.
        arange = torch.arange(seq_length, device=self.device).unsqueeze(0)     # [1, seq_length]
        raw_frame = starts.unsqueeze(1) + arange                                # [num_slices, seq_length]
        # Circular wrap for short motions: frame % (T-1) keeps within
        # valid obs range [0, T-2] (T-1 is next_obs of last transition).
        usable_lens = (motion_lens - 1).clamp_min(1).unsqueeze(1)              # [num_slices, 1]
        frame_cur = (raw_frame % usable_lens).reshape(-1)                       # [B]
        frame_nxt = ((raw_frame + 1) % usable_lens).reshape(-1)                 # [B]
        motion_flat = motion_picks.unsqueeze(1).expand(-1, seq_length).reshape(-1)  # [B]

        # Fully-vectorized gather using the flat obs buffers (one big
        # indexed read per leaf tensor instead of a Python loop over
        # unique motions). Scales to O(1) in the number of motions —
        # matters a lot when the dataset is clip-diced (862 motions
        # instead of 40) and ``num_slices`` would otherwise touch ~128
        # distinct motions per ``sample()`` call, 16 times per iter.
        global_cur = self._motion_obs_starts[motion_flat] + frame_cur  # [B]
        global_nxt = self._motion_obs_starts[motion_flat] + frame_nxt  # [B]
        out_state = self._flat_state[global_cur]
        out_priv = self._flat_priv[global_cur]
        out_act = self._flat_last_action[global_cur]
        out_hist = self._flat_history_actor[global_cur]
        out_state_n = self._flat_state[global_nxt]
        out_priv_n = self._flat_priv[global_nxt]
        out_act_n = self._flat_last_action[global_nxt]
        out_hist_n = self._flat_history_actor[global_nxt]

        B = out_state.shape[0]
        terminated = torch.zeros((B, 1), dtype=torch.bool, device=self.device)
        z_dummy = torch.zeros((B, 0), device=self.device)   # filled by caller if needed
        # The agent signature expects a z entry; we provide zeros of a
        # caller-determined dim. We cannot know z_dim here, so we leave it
        # zero-width and let the caller broadcast / fill. The BFM reference
        # implementation uses the agent's own z-sampler to overwrite this
        # before the discriminator step.

        return {
            "observation": {
                "state": out_state,
                "privileged_state": out_priv,
                "last_action": out_act,
                "history_actor": out_hist,
            },
            "action": out_act,   # reconstructed "last_action" doubles as the demo action
            "z": z_dummy,
            "next": {
                "observation": {
                    "state": out_state_n,
                    "privileged_state": out_priv_n,
                    "last_action": out_act_n,
                    "history_actor": out_hist_n,
                },
                "terminated": terminated,
            },
        }

    @torch.no_grad()
    def sample_chunks(self, batch_size: int, num_chunks: int,
                      target_device: str | torch.device,
                      seq_length: int | None = None) -> list[dict]:
        """Sample ``num_chunks`` batches of size ``batch_size`` in ONE call.

        Each chunk preserves the ``[N x seq_length]`` ordering ``sample()``
        produces (the agent's ``encode_expert`` relies on it), because we
        concatenate batches along the row axis and slice contiguously.
        """
        seq_length = int(seq_length) if seq_length is not None else self.seq_length
        # Round each chunk's batch down to a multiple of seq_length.
        batch_size = max(seq_length, (batch_size // seq_length) * seq_length)

        def _move(x: torch.Tensor) -> torch.Tensor:
            return x.to(target_device, non_blocking=True) if x.device != torch.device(target_device) else x

        # Single fused sample for all chunks: one motion/frame draw, one
        # gather per leaf, then slice. Eliminates the num_chunks-sized
        # Python loop of sample() calls (the hot path when the dataset
        # is clip-diced — ~862 motions, with hundreds of unique picks
        # per batch). Keeps the N × seq_length contiguous layout.
        big = self.sample(batch_size * num_chunks, seq_length=seq_length)

        obs_keys = ("state", "privileged_state", "last_action", "history_actor")
        obs_flat = {k: _move(big["observation"][k]) for k in obs_keys}
        next_obs_flat = {k: _move(big["next"]["observation"][k]) for k in obs_keys}
        action_flat = _move(big["action"])
        z_flat = _move(big["z"])
        term_flat = _move(big["next"]["terminated"])

        chunks: list[dict] = []
        for i in range(num_chunks):
            s = slice(i * batch_size, (i + 1) * batch_size)
            chunks.append({
                "observation": {k: obs_flat[k][s] for k in obs_keys},
                "action": action_flat[s],
                "z": z_flat[s],
                "next": {
                    "observation": {k: next_obs_flat[k][s] for k in obs_keys},
                    "terminated": term_flat[s],
                },
            })
        return chunks


# ---------------------------------------------------------------------------
# 3. ZBuffer -- rolling reservoir of recent z vectors
# ---------------------------------------------------------------------------

class ZBuffer:
    """Circular buffer of recently-seen z vectors for mix-rollout sampling."""

    def __init__(self, capacity: int, z_dim: int, device: str | torch.device = "cpu") -> None:
        self.capacity = int(capacity)
        self.z_dim = int(z_dim)
        self.device = torch.device(device)
        self._storage = torch.zeros((self.capacity, self.z_dim), device=self.device)
        self._idx = 0
        self._is_full = False

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._idx

    def empty(self) -> bool:
        return self._idx == 0 and not self._is_full

    @torch.no_grad()
    def add(self, z_batch: torch.Tensor) -> None:
        z_batch = z_batch.to(self.device, non_blocking=True)
        n = z_batch.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            # Keep the most recent ``capacity`` samples.
            self._storage.copy_(z_batch[-self.capacity:])
            self._is_full = True
            self._idx = 0
            return
        end = self._idx + n
        if end <= self.capacity:
            self._storage[self._idx:end] = z_batch
            self._idx = end
            if self._idx == self.capacity:
                self._is_full = True
                self._idx = 0
        else:
            first = self.capacity - self._idx
            self._storage[self._idx:] = z_batch[:first]
            self._storage[: n - first] = z_batch[first:]
            self._is_full = True
            self._idx = n - first

    @torch.no_grad()
    def sample(self, batch_size: int, device: str | torch.device | None = None) -> torch.Tensor:
        n = len(self)
        if n == 0:
            raise RuntimeError("ZBuffer.sample() called on empty buffer")
        idx = np.random.randint(0, n, size=batch_size)
        idx_t = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        out = self._storage[idx_t].clone()
        if device is not None:
            out = out.to(torch.device(device))
        return out
