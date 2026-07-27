# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner for BFM-Zero's FB-CPR-Aux algorithm.

Orchestrates the rollout -> replay -> update loop as in BFM-Zero's ``train.py``,
but using our Isaac Lab / rsl_rl conventions. The env must expose a
**dict observation** with keys ``{state, privileged_state, last_action,
history_actor}`` (and must forward per-step aux_rewards in ``extras["aux_rewards"]``).
"""

from __future__ import annotations

import math
import os
import shutil
import statistics
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import torch

from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

from isaaclab_rl.rsl_rl.algorithms.fb_cpr import (
    FBCprAux,
    FBCprAuxAlgorithmCfg,
    FBCprCond,
    FBCprCondAlgorithmCfg,
)
from isaaclab_rl.rsl_rl.algorithms.fb_cpr_anchored import AnchoredFBCprAux
from isaaclab_rl.rsl_rl.modules.fb_cpr_policy import (
    FBCprAuxPolicy,
    FBCprCondNetworkCfg,
    FBCprCondPolicy,
    FBCprNetworkCfg,
)
from isaaclab_rl.rsl_rl.storage.fb_cpr_storage import (
    FBCprExpertBuffer,
    FBCprReplayBuffer,
)

__all__ = ["FBCprRunner", "FBCprCondRunner", "AnchoredFBCprRunner"]


def _replay_sibling_path(ckpt_path: str) -> str:
    """``/a/b/model_<n>.pt`` -> ``/a/b/model_<n>.replay.pt`` (keeps the
    light policy/optimizer ckpt separate from the big replay state).
    Works for any ``.pt`` filename; appends ``.replay.pt`` if the input
    has no ``.pt`` extension.
    """
    if ckpt_path.endswith(".pt"):
        return ckpt_path[:-3] + ".replay.pt"
    return ckpt_path + ".replay.pt"


class _PrefetchedSampler:
    """Tiny iterator shim that exposes a ``.sample(batch_size)`` method backed
    by a pre-sampled list of chunks.

    Used by :class:`FBCprRunner` to batch the N×CPU→GPU transfers that would
    otherwise happen inside the algorithm's per-update ``replay_buffer[...]\
    .sample()`` calls.
    """

    def __init__(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        self._cursor = 0

    def peek(self, key: str, default=None):
        if self._cursor >= len(self._chunks):
            return default
        return self._chunks[self._cursor].get(key, default)

    def sample(self, batch_size: int, *args, **kwargs) -> dict:
        if self._cursor >= len(self._chunks):
            raise RuntimeError(
                "_PrefetchedSampler exhausted — prefetch size was too small."
            )
        chunk = self._chunks[self._cursor]
        self._cursor += 1
        return chunk


class FBCprRunner:
    """Minimal BFM-Zero-style training runner.

    Responsibilities:
      * Build policy + algorithm + replay + expert buffers.
      * Drive env-step rollouts with the per-env z context.
      * Write transitions into the replay buffer.
      * Trigger ``num_agent_updates`` gradient updates every
        ``update_agent_every`` env steps past ``num_seed_steps``.
      * Save/load checkpoints.

    NOTE: This runner does not subclass BaseRunner. BaseRunner assumes a
    single flat obs tensor; BFM-Zero is dict-obs through-and-through.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cuda:0",
        **kwargs,
    ) -> None:
        self.cfg = train_cfg
        self.alg_cfg: dict = train_cfg["algorithm"]
        self.policy_cfg: dict = train_cfg["policy"]
        self.device = device
        self.env = env
        self.env_unwrapped = env.unwrapped  # type: ignore
        self.log_dir = log_dir
        # Resume bookkeeping for the initial parameter synchronization in
        # learn(). train.py calls load() on every rank, so an exact checkpoint
        # restore makes the subsequent full-policy broadcast redundant. Missing
        # or explicitly reinitialized parameters still require rank-0 sync.
        self._checkpoint_loaded = False
        self._checkpoint_requires_parameter_sync = False
        self._is_resumed_run = False

        # Scrub class_name keys (consumed before network instantiation).
        self.policy_cfg.pop("class_name", None)
        self.alg_cfg.pop("class_name", None)

        # Multi-GPU bookkeeping (matches rsl_rl OnPolicyRunner._configure_multi_gpu).
        self.gpu_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.environ.get("RANK", "0"))
        self.gpu_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        # Initialize the default process group so ``torch.distributed.barrier()``
        # and our ``broadcast_parameters`` / ``reduce_gradients`` calls work.
        # train.py relies on this happening inside the runner constructor, the
        # same contract OnPolicyRunner honours.
        if self.is_distributed and not torch.distributed.is_initialized():
            # Bind PyTorch's current device BEFORE init_process_group so NCCL's
            # lazy comm creation at the first collective lands on the rank-local
            # GPU. When CUDA_VISIBLE_DEVICES is masked to one GPU (our launch
            # script sets it to $LOCAL_RANK), the rank-local GPU is index 0
            # inside this process; otherwise fall back to local_rank.
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            _dev_idx = 0 if (cvd and "," not in cvd) else self.gpu_local_rank
            torch.cuda.set_device(_dev_idx)
            torch.distributed.init_process_group(
                backend="nccl",
                rank=self.gpu_global_rank,
                world_size=self.gpu_world_size,
                device_id=torch.device(f"cuda:{_dev_idx}"),
            )

        # Seed everything (match BFM's ``set_seed_everywhere``). This fixes
        # the "NaN one run, clean the next" randomness we were seeing from
        # an unseeded PyTorch/NumPy/Python-random triple.
        seed = int(self.cfg.get("seed", 42)) + self.gpu_global_rank
        import random
        import numpy as np
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"[FBCprRunner] set_seed_everywhere(seed={seed})", flush=True)

        # Resolve observation space from the env. The env emits a flat
        # ``policy`` tensor (and ``critic``) with per-term slices in
        # ``policy_meta`` / ``critic_meta``. We build a Dict observation
        # space that groups per-term slices into the 4 BFM-agent keys
        # (``state``, ``privileged_state``, ``last_action``,
        # ``history_actor``) and memoize the slice ranges once so every
        # env.step output can be converted into a dict in one pass.
        self.action_dim = int(self.env_unwrapped.single_action_space.shape[0])
        self._obs_key_groups = self._resolve_obs_key_groups()
        self.obs_space = self._build_dict_obs_space()

        # --- Policy -----------------------------------------------------
        net_cfg = self._build_network_cfg(self.policy_cfg)
        self.policy = self._POLICY_CLS(self.obs_space, action_dim=self.action_dim, cfg=net_cfg)

        # --- Algorithm --------------------------------------------------
        # Optionally tie the training batch_size to num_envs (per-rank). Must be
        # set BEFORE _build_algo_cfg -> the algorithm bakes batch_size into the
        # FB off-diagonal mask (torch.eye(batch_size)) and the LR scaling at
        # construction, so overriding it later would shape-mismatch. Opt-in via
        # ``batch_size_eq_num_envs``; default keeps the configured batch_size.
        if bool(self.alg_cfg.get("batch_size_eq_num_envs", False)):
            _bs = int(self.env.num_envs)
            _old_bs = int(self.alg_cfg.get("batch_size", 1024))
            self.alg_cfg["batch_size"] = _bs
            print(f"[FBCprRunner] batch_size tied to num_envs: {_old_bs} -> {_bs} "
                  f"(per rank).", flush=True)
        algo_cfg = self._build_algo_cfg(self.alg_cfg)
        self.alg = self._ALGO_CLS(self.policy, cfg=algo_cfg, device=self.device)

        # --- Expert buffer ---------------------------------------------
        expert_path = self.alg_cfg.get("expert_dataset_path")
        expert_device = self.alg_cfg.get("expert_dataset_device", "cuda")
        # Resolve bare "cuda" to the rank-local visible GPU. With
        # CUDA_VISIBLE_DEVICES masked the rank-local GPU is index 0 inside
        # this process; otherwise it's local_rank.
        if expert_device == "cuda":
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            _exp_idx = 0 if (cvd and "," not in cvd) else self.gpu_local_rank
            expert_device = f"cuda:{_exp_idx}"
        # The one-time load-time FK compose runs on GPU (fast) even when the
        # buffer is STORED on CPU (expert_dataset_device="cpu" — the big dataset
        # off the VRAM-constrained GPU). Resolve a GPU for compose; when the
        # buffer is already on GPU, compose there too (compose_device == device).
        if str(expert_device).startswith("cpu"):
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            _cmp_idx = 0 if (cvd and "," not in cvd) else self.gpu_local_rank
            expert_compose_device = f"cuda:{_cmp_idx}" if torch.cuda.is_available() else "cpu"
        else:
            expert_compose_device = expert_device
        distributed_expert = bool(self.alg_cfg.get("distributed_expert", False))
        tracking_bin_seconds = float(
            self.alg_cfg.get("tracking_failure_bin_size_s", 0.0)
        )
        tracking_step_dt = float(
            getattr(self.env.unwrapped, "step_dt", 1.0 / 50.0)
        )
        tracking_bin_frames = (
            max(1, int(round(tracking_bin_seconds / tracking_step_dt)))
            if tracking_bin_seconds > 0.0
            else 0
        )
        self.expert_buffer = FBCprExpertBuffer(
            pt_path=expert_path,
            seq_length=net_cfg.seq_length,
            device=expert_device,
            length_proportional_priors=bool(
                self.alg_cfg.get("length_proportional_priors", True)
            ),
            distributed_shard=distributed_expert and self.is_distributed,
            shard_rank=self.gpu_global_rank,
            shard_world_size=self.gpu_world_size,
            shard_seed=int(self.alg_cfg.get("expert_shard_seed", 42)),
            # Optional keypoint-list override (BFM-One: 26-body priv). When
            # None the buffer falls back to the precompute script's
            # KEYPOINT_NAMES (31-body). Must match the env's
            # priv_max_local_self body_names so B sees the same layout.
            keypoint_names=self.alg_cfg.get("expert_keypoint_names", None),
            # Anchored variant: emit expert anchored_pose for the spatial CPR
            # discriminator (anchor = each motion's first frame).
            emit_anchored_pose=bool(self.alg_cfg.get("store_world_pose", False)),
            anchored_pose_clamp=float(self.alg_cfg.get("anchor_pose_clamp", 10.0)),
            # Match the policy's p_A so expert & policy z_spatial share a
            # distribution (spatial disc must judge motion, not z-region).
            anchor_alpha_gt=float(self.alg_cfg.get("anchor_alpha_gt", 0.34)),
            anchor_random_xy_range=float(self.alg_cfg.get("anchor_random_xy_range", 10.0)),
            # Reframe expert priv body-pose into the anchor frame (matches the
            # env's anchor_frame_body=True). Same flag drives both sides.
            anchor_frame_body=bool(self.alg_cfg.get("anchor_frame_body", False)),
            # Append a heading-frame body tail to composed priv (matches the
            # env's _obs_max_local_self include_heading_body=True). Same flag
            # drives env + expert so the priv layout (24K-5) agrees.
            priv_include_heading_body=bool(
                self.alg_cfg.get("priv_include_heading_body", False)),
            # Compose the expert history_actor at the SAME history length the env
            # uses, so its dim matches the env obs-normalizer BatchNorm. Derive H
            # directly from the env's history_actor obs dim (= H * 93) rather than
            # from actor_arch: the deepened (H=9) env may be paired with EITHER the
            # transformer OR the MLP actor (e.g. the BFM-0.5 mlp-actor bisection),
            # and both need the expert history at the env's H. None -> dataset
            # default only when the env carries no history_actor group.
            history_len_override=self._expert_history_len_override(),
            # FK compose on GPU even when stored on CPU (see above).
            compose_device=expert_compose_device,
            expert_tracking_circular_wrap=bool(
                self.alg_cfg.get("expert_tracking_circular_wrap", False)
            ),
            tracking_failure_bin_frames=tracking_bin_frames,
        )
        # Forward to the env so RSI can pull from it.
        if hasattr(self.env_unwrapped, "set_expert_buffer"):
            self.env_unwrapped.set_expert_buffer(self.expert_buffer)

        # --- Replay buffer ---------------------------------------------
        aux_reward_names = list(algo_cfg.aux_rewards_scaling.keys())
        # Anchored variant (Global-through-Anchoring) needs the world-frame
        # SE(2) root pose per transition so anchor-relabeling can transform
        # g_t by an arbitrary A at update time. Gated on a cfg flag so the
        # standard tasks store nothing extra.
        self._store_world_pose = bool(self.alg_cfg.get("store_world_pose", False))
        extra_field_shapes = (
            {"root_xy": (2,), "root_yaw": (1,)} if self._store_world_pose else None
        )
        # history_actor recompose-on-sample (memory saving). When enabled, the
        # replay stores only the newest per-step history frame and rebuilds the
        # full [H*93] window on sample (byte-exact; see FBCprReplayBuffer). Off
        # by default; the spec (H + per-term block widths) is derived from the
        # env's history_actor obs dim so it self-adapts to H.
        history_recompose = self._history_recompose_spec()
        self.replay_buffer = FBCprReplayBuffer(
            capacity=int(self.alg_cfg.get("replay_capacity", 5_120_000)),
            num_envs=self.env.num_envs,
            obs_space=self.obs_space,
            action_dim=self.action_dim,
            z_dim=self.policy.z_dim,
            aux_reward_names=aux_reward_names,
            device=self.alg_cfg.get("replay_device", "cpu"),
            extra_field_shapes=extra_field_shapes,
            # Transformer actor: ask the buffer to also return the H+1 timestep
            # window for the parallel actor loss (0 = off / MLP actor). Derived
            # from the policy's actor_history_len when actor_arch=="transformer".
            actor_window_len=self._actor_window_len(),
            history_recompose=history_recompose,
            replay_sampling_mode=str(
                self.alg_cfg.get("replay_sampling_mode", "uniform_transition")
            ),
        )

        # --- Seed / rhythm controls ------------------------------------
        self.num_seed_steps = int(self.alg_cfg.get("num_seed_steps", 10_240))
        # Per-rank local-step threshold below which agent UPDATES are held off
        # (buffer still fills via rollout). Normally == num_seed_steps; raised on
        # a no-replay resume (see load()) so the resumed policy refills the empty
        # buffer on-policy before updates begin. The random-action warmup is
        # gated separately by num_seed_steps.
        self._delay_updates_until = self.num_seed_steps
        # Per-rank local-step threshold below which the rollout uses UNIFORM-
        # RANDOM actions (== num_seed_steps on a fresh run, where the policy is
        # untrained). On a no-replay resume the policy is already trained, so
        # this is set to 0 (see load()) — the refill collection is fully on-
        # policy, not random.
        self._random_seed_until = self.num_seed_steps
        self.num_agent_updates = int(self.alg_cfg.get("num_agent_updates", 16))
        self.update_agent_every = int(self.alg_cfg.get("update_agent_every", 1024))
        self.save_interval = int(self.cfg.get("save_interval", 50))

        # --- Per-env exploration-std gradient (BEHAVIOR only) ----------
        # Each env rolls out with its own exploration std, drawn per episode
        # uniformly in [explore_std_min, explore_std_max], giving the replay a
        # gradient of exploration scales (better (s,a) coverage -> easier actor
        # extraction). The TD target / actor objective still use the FIXED
        # cfg.actor_std, so Q fits a single well-defined policy (the per-env std
        # only diversifies data collection, not the value target). Disabled
        # (falls back to scalar actor_std) when min==max or the knobs are unset.
        self.explore_std_min = float(self.alg_cfg.get("explore_std_min", 0.0))
        self.explore_std_max = float(self.alg_cfg.get("explore_std_max", 0.0))
        self._use_explore_std_grad = self.explore_std_max > self.explore_std_min > 0.0
        self._explore_std: torch.Tensor | None = None

        # --- Async S3 checkpoint mirror ------------------------------- #
        # If set, every light checkpoint is uploaded (head rank only) to this
        # S3 prefix in a BACKGROUND daemon thread — non-blocking, training
        # never waits on the network and never crashes on upload failure. We
        # maintain ONE rolling object (overwrite the same key each time) so S3
        # holds the latest checkpoint, not a growing pile. Configurable via
        #   cfg.s3_ckpt_uri  = "s3://bucket/prefix"   (None/"" disables)
        #   cfg.s3_ckpt_name = "model_latest.pt"      (the single rolling key)
        # Env override: BFM_S3_CKPT_URI takes precedence over the cfg value.
        self.s3_ckpt_uri = str(
            os.environ.get("BFM_S3_CKPT_URI", self.cfg.get("s3_ckpt_uri", "")) or "").rstrip("/")
        self.s3_ckpt_name = str(self.cfg.get("s3_ckpt_name", "model_latest.pt"))
        self._s3_thread: threading.Thread | None = None
        if self.s3_ckpt_uri and shutil.which("aws") is None:
            print("[FBCprRunner] WARN: s3_ckpt_uri set but 'aws' CLI not found — "
                  "S3 checkpoint mirroring disabled.", flush=True)
            self.s3_ckpt_uri = ""

        # --- Tracking-eval schedule (BFM-style) ---------------------- #
        # ``eval_every_steps``: env-step interval between evals (0 = off).
        #   BFM production: 9_600_000.
        # ``eval_rollout_length``: per-motion tracking rollout length. BFM
        #   pins this to the longest motion in the batch; we cap at a fixed
        #   value so eval cost is bounded.
        # ``eval_update_priorities``: if True, feed MPJPE-based weights
        #   back into the expert buffer (prioritized RSI sampling).
        # ``eval_priority_*``: clamp + scale for the feedback (BFM uses
        #   exp mode scale=2, min=0.5, max=2.0).
        # NO initial eval: ``_last_eval_step`` starts at 0 so the first
        # eval only fires once ``tot_timesteps - 0 >= eval_every_steps``.
        self.eval_every_steps = int(self.alg_cfg.get("eval_every_steps", 9_600_000))
        self.eval_rollout_length = int(self.alg_cfg.get("eval_rollout_length", 250))
        self.eval_emd_workers = max(1, int(self.alg_cfg.get("eval_emd_workers", 4)))
        self.eval_update_priorities = bool(self.alg_cfg.get("eval_update_priorities", True))
        self.eval_priority_min = float(self.alg_cfg.get("eval_priority_min", 0.5))
        self.eval_priority_max = float(self.alg_cfg.get("eval_priority_max", 2.0))
        self.eval_priority_scale = float(self.alg_cfg.get("eval_priority_scale", 2.0))
        self.eval_priority_mode = str(self.alg_cfg.get("eval_priority_mode", "exp"))
        self._last_eval_step = 0

        # --- Logging ---------------------------------------------------
        self.num_steps_per_env = int(self.cfg.get("num_steps_per_env", 1))
        log_ws_cap = int(self.cfg.get("log_env_steps_world_size_cap", 0))
        self._log_world_size = (
            min(self.gpu_world_size, log_ws_cap)
            if log_ws_cap > 0
            else self.gpu_world_size
        )
        self.writer = None
        self.tot_timesteps = 0
        # Kept separate from tot_timesteps so logging normalization cannot
        # alter LR schedules or other training behavior.
        self.log_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0

    # --- BFM-Zero obs-space composition ------------------------------- #

    # Maps the BFM-agent obs keys to the per-term names we expect to find
    # in the env's observation_cfg. The env must define obs terms with
    # these names; the runner concatenates them in the same order inside
    # each group to match the expert-dataset layout.
    #
    # Tasks extending BFM-Zero with new obs keys (e.g. BFM-Terrain's
    # ``height_scan``) should set the algorithm-cfg entry
    # ``obs_key_groups`` to override the default. The env's obs cfg must
    # have matching term names.
    _BFM_KEY_GROUPS_DEFAULT: dict[str, tuple[str, ...]] = {
        "state": ("state", "gravity", "root_ang_vel"),
        "last_action": ("last_action",),
        # History fields are concatenated in alphabetical order to match BFM's
        # ``_get_obs_history_actor`` sorted-key iteration:
        # ``[actions, base_ang_vel, dof_pos, dof_vel, projected_gravity]``.
        "history_actor": (
            "history_actions",
            "history_base_ang_vel",
            "history_dof_pos",
            "history_dof_vel",
            "history_projected_gravity",
        ),
        "privileged_state": ("priv_max_local_self",),
    }

    @property
    def _BFM_KEY_GROUPS(self) -> dict[str, tuple[str, ...]]:
        """Resolve the per-agent-key obs mapping from cfg (``obs_key_groups``),
        with ``_BFM_KEY_GROUPS_DEFAULT`` as fallback.
        """
        override = self.alg_cfg.get("obs_key_groups", None)
        if override:
            # Normalize list -> tuple for stable iteration.
            return {k: tuple(v) for k, v in override.items()}
        return dict(self._BFM_KEY_GROUPS_DEFAULT)

    def _resolve_obs_key_groups(self) -> dict[str, dict[str, slice]]:
        """Compute per-term slice ranges in the flat policy / critic tensors.

        Returns a dict::

            {
                "state":           {"dim": int, "policy_slice": slice, "critic_slice": slice},
                "privileged_state": {...},
                "last_action":      {...},
                "history_actor":    {...},
            }

        Each entry describes where to find the concatenated per-term outputs
        that compose that BFM-agent key inside the env's flat obs tensors.
        """
        obs_cfg = self.env_unwrapped.main_observation_cfg
        policy_keys: list[str] = obs_cfg.policy_obs_keys
        critic_keys: list[str] = obs_cfg.critic_obs_keys
        # Probe term widths by evaluating once (the env has already built
        # the observation_terms inside _setup_env).
        probe = obs_cfg.compute(compute_critic=True, compute_meta=False)
        policy_tensor: torch.Tensor = probe["policy"]
        critic_tensor: torch.Tensor = probe["critic"]
        # Slice widths per term — derive from the dict returned above by
        # re-running the per-term forward to get shapes.
        term_widths: dict[str, int] = {}
        policy_cursor = 0
        critic_cursor = 0
        policy_slices: dict[str, slice] = {}
        critic_slices: dict[str, slice] = {}
        for key in policy_keys:
            # The term returns (policy_obs, critic_obs) both of shape
            # [B, width] after the .view() in ObservationCfg.compute().
            p_obs, c_obs = obs_cfg.observation_terms[key]()
            p_w = p_obs.view(p_obs.shape[0], -1).shape[1]
            c_w = c_obs.view(c_obs.shape[0], -1).shape[1]
            term_widths[key] = p_w
            policy_slices[key] = slice(policy_cursor, policy_cursor + p_w)
            critic_slices[key] = slice(critic_cursor, critic_cursor + c_w)
            policy_cursor += p_w
            critic_cursor += c_w
        for key in critic_keys:
            _, c_obs = obs_cfg.observation_terms[key]()
            c_w = c_obs.view(c_obs.shape[0], -1).shape[1]
            term_widths[key] = c_w
            critic_slices[key] = slice(critic_cursor, critic_cursor + c_w)
            critic_cursor += c_w

        # Now carve out the 4 BFM-agent groups.
        groups: dict[str, dict] = {}
        for group_name, term_names in self._BFM_KEY_GROUPS.items():
            # Determine which flat tensor the group lives in.
            # - ``privileged_state`` terms live in critic-only slots.
            # - All other groups (``state`` / ``last_action`` / ``history_actor``)
            #   live in the policy tensor.
            from_critic = all(t.startswith("priv_") for t in term_names)
            dim = 0
            start = None
            prev_end = None
            for t in term_names:
                if t not in term_widths:
                    raise KeyError(
                        f"BFM-Zero runner expected obs term '{t}' in the env's observation_cfg "
                        f"for BFM-agent key '{group_name}', but it was not found. "
                        f"Available terms: {list(term_widths.keys())}"
                    )
                sl = critic_slices[t] if from_critic else policy_slices[t]
                if prev_end is not None and sl.start != prev_end:
                    raise AssertionError(
                        f"BFM-Zero obs terms {term_names} for group '{group_name}' are not "
                        f"contiguous in the flat tensor (term '{t}' starts at {sl.start}, "
                        f"but previous term ended at {prev_end}). Order them contiguously in "
                        f"the env's observation cfg."
                    )
                if start is None:
                    start = sl.start
                prev_end = sl.stop
                dim += term_widths[t]
            end = start + dim
            groups[group_name] = {
                "dim": dim,
                "from_critic": from_critic,
                "slice": slice(start, end),
            }
        return groups

    def _build_dict_obs_space(self):
        import gymnasium as gym
        import numpy as np
        spaces = {}
        for key, meta in self._obs_key_groups.items():
            d = meta["dim"]
            spaces[key] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32
            )
        return gym.spaces.Dict(spaces)

    def _flat_to_dict(self, policy_flat: torch.Tensor, critic_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for key, meta in self._obs_key_groups.items():
            src = critic_flat if meta["from_critic"] else policy_flat
            out[key] = src[:, meta["slice"]].to(self.device, non_blocking=True)
        return out

    # --- helpers to translate rl_cfg.py configclass dict -> dataclasses -- #

    # --- class-level refs so subclasses can swap the policy / algorithm /
    # cfg classes without duplicating __init__ ---
    _POLICY_CLS = FBCprAuxPolicy
    _ALGO_CLS = FBCprAux
    _NET_CFG_CLS = FBCprNetworkCfg
    _ALGO_CFG_CLS = FBCprAuxAlgorithmCfg

    @classmethod
    def _build_network_cfg(cls, policy_cfg: dict) -> FBCprNetworkCfg:
        cfg = cls._NET_CFG_CLS()
        for k, v in policy_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        for tuple_key in [
            "backward_input_keys",
            "forward_input_keys",
            "actor_input_keys",
            "critic_input_keys",
            "aux_critic_input_keys",
            "discriminator_input_keys",
        ]:
            val = getattr(cfg, tuple_key)
            if isinstance(val, list):
                setattr(cfg, tuple_key, tuple(val))
        return cfg

    @classmethod
    def _build_algo_cfg(cls, alg_cfg: dict) -> FBCprAuxAlgorithmCfg:
        cfg = cls._ALGO_CFG_CLS(aux_rewards_scaling=dict(alg_cfg.get("aux_rewards_scaling", {})))
        for k, v in alg_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def _actor_window_len(self) -> int:
        """H (number of PAST frames) the buffer must gather for the transformer
        actor's parallel loss; the window is H+1 (offsets -H..0). Equals the
        policy's actor_history_len when actor_arch=="transformer", else 0 (MLP)."""
        if str(self.policy_cfg.get("actor_arch", "mlp")) != "transformer":
            return 0
        return int(self.policy_cfg.get("actor_history_len", 9))

    # Per-frame block layout of the ``history_actor`` obs, in ENV STORAGE ORDER
    # (the flat blob is per-block-major, each block frame-major newest-first):
    #   history_actions | history_base_ang_vel | history_dof_pos
    #   | history_dof_vel | history_projected_gravity
    # Matches BFMZeroEnvCfg.observations term order. Sum = frame_dim (93).
    _HISTORY_ACTOR_BLOCKS: tuple[tuple[str, int], ...] = (
        ("act", 29), ("angv", 3), ("dofp", 29), ("dofv", 29), ("grav", 3),
    )

    def _history_recompose_spec(self) -> dict | None:
        """Build the replay ``history_recompose`` spec, or None if disabled.

        Enabled by ``alg_cfg['recompose_history_actor']`` (default False). Stores
        only the newest history frame and rebuilds the full window on sample;
        byte-exact to the env's blob (verified in test_history_recompose.py).
        H is derived from the env's history_actor obs dim / frame_dim so it
        self-adapts. Requires the MLP actor (the transformer actor-window path
        re-reads full-width history — the buffer raises if both are set)."""
        if not bool(self.alg_cfg.get("recompose_history_actor", False)):
            return None
        ha = self.obs_space.spaces.get("history_actor", None) if hasattr(self.obs_space, "spaces") else None
        if ha is None:
            print("[FBCprRunner] recompose_history_actor set but no history_actor "
                  "obs group — disabling recompose.", flush=True)
            return None
        blocks = list(self._HISTORY_ACTOR_BLOCKS)
        frame_dim = sum(w for _, w in blocks)
        dim = int(ha.shape[0])
        if dim % frame_dim != 0:
            raise ValueError(
                f"recompose_history_actor: history_actor dim {dim} is not a "
                f"multiple of frame_dim {frame_dim} (blocks {blocks}). The env's "
                f"history term widths must match _HISTORY_ACTOR_BLOCKS."
            )
        H = dim // frame_dim
        print(f"[FBCprRunner] history_actor recompose-on-sample ENABLED "
              f"(H={H}, frame_dim={frame_dim}, storing 1 frame vs {H} -> "
              f"replay history_actor {H}x smaller).", flush=True)
        return {"key": "history_actor", "H": H, "blocks": blocks}

    def _expert_history_len_override(self) -> int | None:
        """H at which to compose the expert ``history_actor`` so its dim matches
        the env obs-normalizer BatchNorm. Derived from the env's actual
        history_actor obs dim (= H * frame_dim) so it is correct for ANY actor
        (transformer or MLP) running on a deepened-history env. Returns None when
        the env carries no history_actor group (dataset default applies)."""
        ha = self.obs_space.spaces.get("history_actor", None) if hasattr(self.obs_space, "spaces") else None
        if ha is None:
            return None
        frame_dim = int(self.policy_cfg.get("actor_frame_dim", 93))
        dim = int(ha.shape[0])
        if frame_dim <= 0 or dim % frame_dim != 0:
            return None
        return dim // frame_dim

    def _sample_explore_std(self, n: int) -> torch.Tensor:
        """``n`` per-env exploration stds, uniform in [explore_std_min,
        explore_std_max]. Behavior-policy only; the TD target uses actor_std."""
        lo, hi = self.explore_std_min, self.explore_std_max
        return torch.rand(n, 1, device=self.device) * (hi - lo) + lo

    def _apply_tracking_sim_anchor(self, robot) -> None:
        """Two-frame anchor: set the env ``anchored_pose`` anchor for the freshly
        resampled tracking envs to the DISPLACED sim pose A_init·A_anchor, in
        WORLD. A_init = the LIVE (post-reset) robot pose of each tracking env;
        A_anchor = the init-local offset the algo sampled this resample
        (``_tracking_anchor_canon_{xy,yaw}``). The motion-space counterpart
        A^m_init·A_anchor was used to encode the tracking-z, so the actor's
        ``anchored_pose`` obs and the z now live in the SAME frame and the
        implicit reward ⟨B(s),z⟩ aligns (A_anchor=0 -> spawn-anchored). Must be
        called AFTER any terrain RSI reset settles so A_init is the post-reset
        pose. Non-tracking envs and ``_canon_*`` (stored-pose frame) untouched.
        """
        if not (self._store_world_pose and robot is not None
                and hasattr(self.env_unwrapped, "set_anchor")):
            return
        t_ids = getattr(self.alg, "_tracking_env_idx", None)
        aA_xy = getattr(self.alg, "_tracking_anchor_canon_xy", None)
        aA_yaw = getattr(self.alg, "_tracking_anchor_canon_yaw", None)
        if t_ids is None or aA_xy is None or t_ids.numel() == 0:
            return
        # A_init of the tracking envs (live world pose).
        r_xy = robot.data.root_pos_w[t_ids, :2].to(self.device)
        rq = robot.data.root_quat_w[t_ids].to(self.device)
        w_, x_, y_, z_ = rq[:, 0], rq[:, 1], rq[:, 2], rq[:, 3]
        r_yaw = torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
        # A_init · A_anchor: rotate the init-local offset by A_init yaw, translate.
        aA_xy = aA_xy.to(self.device)
        aA_yaw = aA_yaw.to(self.device)
        cy, sy = torch.cos(r_yaw), torch.sin(r_yaw)
        off_x = cy * aA_xy[:, 0] - sy * aA_xy[:, 1]
        off_y = sy * aA_xy[:, 0] + cy * aA_xy[:, 1]
        sim_xy = r_xy + torch.stack([off_x, off_y], dim=-1)
        sim_yaw = r_yaw + aA_yaw
        full_xy = self.env_unwrapped._anchor_xy.clone()
        full_yaw = self.env_unwrapped._anchor_yaw.clone()
        full_xy[t_ids] = sim_xy
        full_yaw[t_ids] = sim_yaw
        self.env_unwrapped.set_anchor(full_xy, full_yaw, env_ids=None)

    # --- training loop ------------------------------------------------- #

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # Only rank 0 logs / prints under DDP (avoid N-way wandb runs and
        # duplicate stdout spam).
        self._is_head = (not self.is_distributed) or (self.gpu_global_rank == 0)

        # Logger init — head rank only.
        if self.log_dir is not None and self.writer is None and self._is_head:
            logger_type = self.cfg.get("logger", "tensorboard").lower()
            if logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            else:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # DDP: broadcast rank-0 parameters + running stats on fresh starts or
        # partial checkpoint restores. On a normal resume, train.py calls load()
        # on every rank, including optimizer state. Re-broadcasting the full XL
        # policy after that restore is both redundant and dangerous: populated
        # Adam state already consumes its steady-state GPU memory, while
        # broadcast_object_list adds a large serialized staging payload.
        #
        # Reduce a one-element eligibility flag first so every rank makes the
        # same skip/broadcast decision. If even one rank did not load, or retained
        # freshly initialized parameters, all ranks take the broadcast path.
        if self.is_distributed:
            can_skip = (
                self._checkpoint_loaded
                and not self._checkpoint_requires_parameter_sync
            )
            skip_flag = torch.tensor(
                int(can_skip), device=self.device, dtype=torch.int32
            )
            torch.distributed.all_reduce(
                skip_flag, op=torch.distributed.ReduceOp.MIN
            )
            if bool(skip_flag.item()):
                print(
                    f"[FBCprRunner] rank {self.gpu_global_rank}: checkpoint "
                    f"restored on every rank; skipping redundant parameter broadcast.",
                    flush=True,
                )
            else:
                print(
                    f"[FBCprRunner] Synchronizing parameters for rank "
                    f"{self.gpu_global_rank}...",
                    flush=True,
                )
                self.alg.broadcast_parameters()

        step_count = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)

        # Per-env z context.
        _robot = self.env.unwrapped.robot if hasattr(self.env.unwrapped, "robot") else None
        _terrain_z_fn = getattr(self.env_unwrapped, "_get_terrain_height_xy", None)
        z_context, terrain_reset = self.alg.maybe_update_rollout_context(
            z=None, step_count=step_count, expert_buffer=self.expert_buffer,
            robot_root_xy=_robot.data.root_pos_w[:, :2].to(self.device) if _robot else None,
            robot_root_quat=_robot.data.root_quat_w.to(self.device) if _robot else None,
            terrain_z_fn=_terrain_z_fn,
        )
        if terrain_reset is not None:
            env_ids = terrain_reset["env_ids"]
            if hasattr(self.env_unwrapped, "_reset_idx"):
                self.env_unwrapped._reset_idx(env_ids)
            self._terrain_rsi_from_tracking(
                env_ids, terrain_reset["motion_ids"], terrain_reset["starts"],
            )
            step_count[env_ids] = 0
            if _robot is not None:
                self.alg.update_tracking_pose_after_reset(
                    env_ids,
                    _robot.data.root_pos_w[:, :2].to(self.device),
                    _robot.data.root_quat_w.to(self.device),
                )

        # Anchored variant: set the initial per-episode anchor for ALL envs to
        # their spawn pose at the start of rollout. We ALSO keep a runner-owned
        # copy of the FIXED spawn origin (``_canon_xy/_canon_yaw``) used to
        # canonicalize the STORED replay pose. The env's ``_anchor`` may later
        # move per-resample (displaced rollout anchor A_init·A_anchor for
        # tracking envs), but ``_canon_*`` stays pinned at spawn so the stored
        # replay frame never jumps mid-episode.
        if (self._store_world_pose and _robot is not None
                and hasattr(self.env_unwrapped, "set_anchor")):
            rq0 = _robot.data.root_quat_w.to(self.device)
            w0, x0, y0, z0 = rq0[:, 0], rq0[:, 1], rq0[:, 2], rq0[:, 3]
            yaw0 = torch.atan2(2 * (w0 * z0 + x0 * y0), 1 - 2 * (y0 * y0 + z0 * z0))
            _xy0 = _robot.data.root_pos_w[:, :2].to(self.device)
            self.env_unwrapped.set_anchor(_xy0, yaw0, env_ids=None)
            self._canon_xy = _xy0.clone()
            self._canon_yaw = yaw0.clone()
            # Apply the displaced sim anchor for the INITIAL tracking window
            # (the z=None resample above already sampled A_anchor) so the first
            # window's actor obs matches its displaced tracking-z. canon_* stays
            # at spawn.
            self._apply_tracking_sim_anchor(_robot)

        # Capture the first policy observation only after all RSI and anchor
        # writes. Observation terms own history buffers, so reading before and
        # after these writes would both return stale data and advance history
        # twice without a simulator step.
        obs_flat, extras = self.env.get_observations()
        obs_dict = self._obs_to_device(obs_flat, extras)

        # Per-env exploration std: init all envs with a fresh draw in
        # [explore_std_min, explore_std_max]. Resampled per episode on done.
        if self._use_explore_std_grad:
            self._explore_std = self._sample_explore_std(self.env.num_envs)

        rewbuffer: deque[float] = deque(maxlen=500)
        lenbuffer: deque[float] = deque(maxlen=500)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_ep_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # BFM-style done tracking. The trajectory buffer stores ALL envs
        # every step (no done-filtering) and uses the ``truncated`` column
        # to mark episode boundaries. ``terminated`` and ``truncated`` from
        # the PREVIOUS step are written alongside the current obs/action.
        prev_terminated = torch.zeros(self.env.num_envs, 1, dtype=torch.bool, device=self.device)
        # A restored replay ends in a different simulator process/state. Mark
        # the first newly appended observations as episode starts so the ring
        # never creates an old-checkpoint -> fresh-simulator transition.
        prev_truncated = torch.full(
            (self.env.num_envs, 1),
            fill_value=len(self.replay_buffer) > 0,
            dtype=torch.bool,
            device=self.device,
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        best_reward = -float("inf")

        steps_since_last_update = 0
        total_metrics: Dict[str, list[torch.Tensor]] = {}
        max_metrics: Dict[str, list[torch.Tensor]] = {}

        # --- Initial (pre-training) tracking eval ---
        # Runs once BEFORE any gradient step so the user sees a baseline
        # MPJPE against the untrained policy. We FORCE ``update_priorities=
        # False`` here — feeding untrained-policy MPJPE back as sampling
        # weights would massively distort early RSI distribution.
        # Skipped when ``eval_every_steps=0`` (eval fully disabled) or on
        # resumed runs (``current_learning_iteration > 0``).
        if (
            self.eval_every_steps > 0
            and self.current_learning_iteration == 0
            and self.eval_rollout_length > 0
        ):
            if self._is_head:
                print(f"[FBCprRunner] running initial tracking eval (no priority update) ...", flush=True)
            eval0 = self._run_tracking_eval(update_priorities=False)
            if self._is_head and eval0 is not None and self.log_dir is not None and self.writer is not None:
                env_steps = int(self.log_timesteps)
                # Log the initial eval on the SAME curve as subsequent
                # evals (``Eval/*``) so wandb/TB show a continuous
                # trajectory from untrained-policy baseline onwards.
                # Priority updates were suppressed above, so feeding this
                # point into the main curve is purely a logging choice.
                for k, v in eval0.items():
                    self.writer.add_scalar(k, v, env_steps)
                print(
                    f"[FBCprRunner] initial eval: "
                    f"mpjpe_mm={eval0.get('Eval/mpjpe_mm', 0):.1f} "
                    f"emd={eval0.get('Eval/emd', 0):.3g} "
                    f"tracking_success={eval0.get('Eval/tracking_success', 0):.3g} "
                    f"num_motions={int(eval0.get('Eval/num_motions', 0))}",
                    flush=True,
                )
            if bool(
                self.alg_cfg.get("rollout_tracking_legacy_schedule", False)
            ):
                # Legacy Gamma0.8 refreshed the complete rollout context after
                # the initial evaluation, including a second tracking assignment
                # draw and any terrain RSI reset it requested.
                z_context, terrain_reset = self.alg.maybe_update_rollout_context(
                    z=None,
                    step_count=step_count,
                    expert_buffer=self.expert_buffer,
                    robot_root_xy=(
                        _robot.data.root_pos_w[:, :2].to(self.device)
                        if _robot else None
                    ),
                    robot_root_quat=(
                        _robot.data.root_quat_w.to(self.device)
                        if _robot else None
                    ),
                    terrain_z_fn=_terrain_z_fn,
                )
                if terrain_reset is not None:
                    env_ids = terrain_reset["env_ids"]
                    if hasattr(self.env_unwrapped, "_reset_idx"):
                        self.env_unwrapped._reset_idx(env_ids)
                    self._terrain_rsi_from_tracking(
                        env_ids,
                        terrain_reset["motion_ids"],
                        terrain_reset["starts"],
                    )
                    step_count[env_ids] = 0
                    if _robot is not None:
                        self.alg.update_tracking_pose_after_reset(
                            env_ids,
                            _robot.data.root_pos_w[:, :2].to(self.device),
                            _robot.data.root_quat_w.to(self.device),
                        )
            # Current scheduling keeps the existing z and cached observation:
            # eval snapshots/restores the environment and does not mutate the
            # rollout context.

        # Track LOCAL env-steps (per-rank) for warmup / update cadence / eval
        # cadence. Each rank has its own replay buffer, so
        # ``num_seed_steps``/``update_agent_every``/``eval_every_steps`` are
        # per-rank budgets. BFM has this too: every rank runs its own train
        # loop. ``self.tot_timesteps`` remains the raw GLOBAL
        # (env_steps x world_size) counter used by LR schedules and checkpoint
        # metadata; ``self.log_timesteps`` is the separately normalized x-axis.
        local_timesteps = getattr(self, "_local_timesteps", 0)

        for it in range(start_iter, tot_iter):
            start = time.time()
            tracking_resamples_before = int(
                getattr(self.alg, "_tracking_resample_count", 0)
            )
            # Per-iteration anchored-goal-following accumulators (tracking envs):
            # sum of |robot_xy - ref_xy| and |wrap(robot_yaw - ref_yaw)| each
            # step AFTER env.step, where ref_* is the reference motion's
            # intrinsic world pose (the anchored global goal). Mean -> Track/
            # global_xy_dev_m, Track/global_yaw_dev_deg. Masked by the tracking-
            # env set (NOT the legacy global-FB-visible flag).
            _trk_xy_sum = 0.0
            _trk_yaw_sum = 0.0
            _trk_count = 0
            rollout_done_count = 0
            rollout_transition_count = 0
            tracking_failure_count = 0
            tracking_bin_success_count = 0
            tracking_failure_eligible_steps = 0
            tracking_failure_stat_chunks: list[torch.Tensor] = []
            # ----- rollout -----
            with torch.inference_mode():
                self.policy.eval()
                for _ in range(self.num_steps_per_env):
                    warmup = local_timesteps < self._random_seed_until
                    if warmup:
                        actions = torch.zeros(
                            self.env.num_envs, self.action_dim, device=self.device
                        ).uniform_(-1.0, 1.0)
                    else:
                        actions = self.policy.act(
                            obs_dict, z_context, mean=False,
                            std=self._explore_std if self._use_explore_std_grad else None)

                    # Anchored variant: capture the CANONICAL (start-relative)
                    # SE(2) root pose NOW — BEFORE env.step — so it matches
                    # ``obs_dict`` (the pre-step obs this transition stores).
                    # We express the world pose g_t in the env's per-episode
                    # SPAWN frame (env._anchor_xy/_anchor_yaw, which the runner
                    # sets to the spawn pose at every reset): g_canon =
                    # [Rot(-s_yaw)(g_xy - s_xy), wrap(g_yaw - s_yaw)]. Storing
                    # start-relative (instead of absolute world) keeps every
                    # transition's pose bounded by per-episode travel (~a few m)
                    # rather than the sim-terrain world (hundreds of m), so the
                    # per-row anchor re-sampling and the cross-row goal-z
                    # re-anchor (preamble) stay in a single, comparable frame
                    # shared with the expert (which self-zeros at its sub-traj
                    # start). Reading after the step would store g_{t+1} on the
                    # g_t obs (off-by-one), corrupting the spatial<->body map.
                    prestep_extras = None
                    if self._store_world_pose and _robot is not None:
                        _rq0 = _robot.data.root_quat_w.to(self.device)
                        _w, _x, _y, _z = _rq0[:, 0], _rq0[:, 1], _rq0[:, 2], _rq0[:, 3]
                        _yaw0 = torch.atan2(2 * (_w * _z + _x * _y),
                                            1 - 2 * (_y * _y + _z * _z))
                        _g_xy = _robot.data.root_pos_w[:, :2].to(self.device)
                        # Canonicalize the STORED replay pose to the runner-owned
                        # FIXED per-episode spawn origin (``_canon_xy/_canon_yaw``),
                        # NOT the env's ``_anchor`` (which now moves per-resample
                        # with the displaced rollout anchor A_init·A_anchor). The
                        # FB relabel re-anchors stored poses itself, so it only
                        # needs ONE stable per-episode frame; decoupling it from
                        # the moving obs anchor keeps the stored frame fixed
                        # across mid-episode resamples (no boundary leak).
                        _s_xy = getattr(self, "_canon_xy", None)
                        _s_yaw = getattr(self, "_canon_yaw", None)
                        if _s_xy is not None and _s_yaw is not None:
                            _d = _g_xy - _s_xy
                            _ca, _sa = torch.cos(-_s_yaw), torch.sin(-_s_yaw)
                            _rel_x = _ca * _d[:, 0] - _sa * _d[:, 1]
                            _rel_y = _sa * _d[:, 0] + _ca * _d[:, 1]
                            _rel_xy = torch.stack([_rel_x, _rel_y], dim=-1)
                            _rel_yaw = torch.atan2(torch.sin(_yaw0 - _s_yaw),
                                                   torch.cos(_yaw0 - _s_yaw))
                        else:
                            # No spawn frame available — fall back to world pose.
                            _rel_xy = _g_xy
                            _rel_yaw = _yaw0
                        prestep_extras = {
                            "root_xy": _rel_xy.clone(),
                            "root_yaw": _rel_yaw.unsqueeze(-1).clone(),
                        }

                    # Push tracking reference positions to env for debug viz.
                    env_u = self.env.unwrapped
                    if hasattr(env_u, "set_ref_motion_keypoints"):
                        ref_pos = self.alg.get_tracking_ref_root_pos(
                            step_count, self.expert_buffer,
                        )
                        if ref_pos is not None:
                            env_u.set_ref_motion_keypoints(ref_pos.unsqueeze(1))

                    # Push global FB targets (XY + yaw) to env before step, and
                    # stash the ref path to score Track/global_*_dev. Skipped when
                    # log_global_track_dev=False (BFM-Zero/0.5: no global goal).
                    _trk_target = None
                    if (bool(self.alg_cfg.get("log_global_track_dev", True))
                            and hasattr(env_u, "set_global_fb_targets")):
                        targets = self.alg.get_global_fb_targets(
                            step_count, self.expert_buffer,
                            robot_root_xy=_robot.data.root_pos_w[:, :2].to(self.device) if _robot else None,
                            robot_root_quat=_robot.data.root_quat_w.to(self.device) if _robot else None,
                        )
                        if targets is not None:
                            xy, yaw, active, t_ids = targets
                            env_u.set_global_fb_targets(xy, yaw, active, tracking_env_ids=t_ids)
                            # Stash the (world) ref-path target + the TRACKING-env
                            # mask to score post-step deviation. NOTE: we mask by
                            # the tracking-env set (t_ids), NOT the ``active``
                            # (global-FB-visible) flag — ``active`` only controls
                            # legacy obs visibility and is mostly off under
                            # global_fb_zero_prob=1.0. ``xy/yaw`` is the
                            # reference motion's intrinsic world pose (heading-
                            # rotated + spawn-offset), i.e. exactly the anchored
                            # global goal the robot should be following.
                            _trk_mask = torch.zeros(
                                self.env.num_envs, dtype=torch.bool, device=self.device)
                            if t_ids is not None and t_ids.numel() > 0:
                                _trk_mask[t_ids] = True
                            _trk_target = (xy, yaw, _trk_mask)

                    # Push whole-body reference (heading-frame priv + joint
                    # pos/vel) for the explicit imitation aux reward.
                    _tracking_ref_priv = None
                    _tracking_ref_joint_pos = None
                    if hasattr(env_u, "set_tracking_ref_whole_body"):
                        wb = self.alg.get_tracking_ref_whole_body(
                            step_count, self.expert_buffer,
                        )
                        if wb is not None:
                            ref_priv, ref_jp, ref_jv, wb_mask = wb
                            _tracking_ref_priv = ref_priv
                            _tracking_ref_joint_pos = ref_jp
                            env_u.set_tracking_ref_whole_body(
                                ref_priv, ref_jp, ref_jv, wb_mask,
                            )

                    new_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    new_obs = self._obs_to_device(new_obs, infos)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # A configurable subset of tracking envs treats sustained
                    # joint-position or pelvis-height error as an episode
                    # boundary. Failed slots draw a failure-prioritized motion
                    # bin, reset to its first frame, and restart a matching z
                    # schedule. The replay row boundary excludes the teleport,
                    # so this remains a continuing-task FB curriculum rather
                    # than an absorbing failure target.
                    failure_info = None
                    if (
                        _tracking_ref_priv is not None
                        and _tracking_ref_joint_pos is not None
                        and "privileged_state" in new_obs
                    ):
                        failure_info = self.alg.get_tracking_failures(
                            env_u.joint_pos,
                            _tracking_ref_joint_pos,
                            new_obs["privileged_state"],
                            _tracking_ref_priv,
                            dones.bool(),
                            self.expert_buffer,
                            terrain_z_fn=_terrain_z_fn,
                        )
                    if failure_info is not None:
                        tracking_bin_success_count += int(
                            failure_info.get("bin_success_count", 0)
                        )
                        enabled = failure_info["enabled"]
                        joint_err = failure_info["joint_mae"]
                        root_h_err = failure_info["root_height_error"]
                        tracking_failure_eligible_steps += int(
                            failure_info["eligible_count"]
                        )
                        tracking_failure_stat_chunks.append(
                            torch.stack(
                                (
                                    joint_err[enabled].sum(),
                                    root_h_err[enabled].sum(),
                                )
                            )
                        )
                        failed_env_ids = failure_info.get("env_ids")
                        if (
                            failed_env_ids is not None
                            and failed_env_ids.numel() > 0
                        ):
                            tracking_failure_count += int(
                                failed_env_ids.numel()
                            )
                            dones = dones.clone()
                            dones[failed_env_ids] = 1
                            history_snapshot = (
                                self._snapshot_observation_histories()
                            )
                            self.env_unwrapped._reset_idx(failed_env_ids)
                            self._terrain_rsi_from_tracking(
                                failed_env_ids,
                                failure_info["motion_ids"],
                                failure_info["reset_frames"],
                                align_to_env_origins=True,
                            )
                            if _robot is not None:
                                self.alg.update_tracking_pose_after_reset(
                                    failed_env_ids,
                                    _robot.data.root_pos_w[:, :2].to(
                                        self.device
                                    ),
                                    _robot.data.root_quat_w.to(self.device),
                                    reset_frames=failure_info[
                                        "reset_frames"
                                    ],
                                    tracking_slots=failure_info["slots"],
                                )
                            fresh_raw = (
                                self.env_unwrapped._get_observations()
                            )
                            self._restore_observation_histories(
                                history_snapshot, failed_env_ids
                            )
                            fresh_obs = self._obs_to_device(fresh_raw)
                            for key in new_obs:
                                if key not in fresh_obs:
                                    continue
                                patched = new_obs[key].clone()
                                patched[failed_env_ids] = fresh_obs[key][
                                    failed_env_ids
                                ]
                                new_obs[key] = patched
                            env_obs = getattr(
                                self.env_unwrapped, "obs_buf", None
                            )
                            if isinstance(env_obs, dict):
                                for key, value in fresh_raw.items():
                                    if key not in env_obs:
                                        continue
                                    env_obs[key][failed_env_ids] = value[
                                        failed_env_ids
                                    ]

                    rollout_done_count += int(dones.bool().sum().item())
                    rollout_transition_count += int(dones.numel())

                    # Anchored-goal-following deviation (tracking envs): robot's
                    # post-step world pose vs the REFERENCE motion's intrinsic
                    # world pose at this step (the anchored global goal). Mean
                    # per iter -> Track/global_xy_dev_m, Track/global_yaw_dev_deg.
                    # This is what the anchored policy should track: does the
                    # robot follow the reference's global path?
                    if _trk_target is not None and _robot is not None:
                        _txy, _tyaw, _tactive = _trk_target
                        _am = _tactive.bool() if _tactive is not None else None
                        if _am is not None and bool(_am.any()):
                            _rxy = _robot.data.root_pos_w[:, :2].to(self.device)
                            _rq = _robot.data.root_quat_w.to(self.device)
                            _ry = torch.atan2(
                                2 * (_rq[:, 0] * _rq[:, 3] + _rq[:, 1] * _rq[:, 2]),
                                1 - 2 * (_rq[:, 2] * _rq[:, 2] + _rq[:, 3] * _rq[:, 3]))
                            _xy_dev = torch.norm(_rxy[_am] - _txy[_am], dim=-1)
                            _yaw_dev = torch.atan2(
                                torch.sin(_ry[_am] - _tyaw[_am]),
                                torch.cos(_ry[_am] - _tyaw[_am])).abs()
                            _trk_xy_sum += float(_xy_dev.sum().item())
                            _trk_yaw_sum += float(_yaw_dev.sum().item())
                            _trk_count += int(_am.sum().item())

                    aux_rewards_dict = self._extract_aux_rewards(infos)
                    # BFM: terminated excludes time_outs. Isaac Lab exposes
                    # time_outs via infos["time_outs"].
                    time_outs = infos.get("time_outs", None)
                    if time_outs is None:
                        time_outs = torch.zeros_like(dones, dtype=torch.bool)
                    elif not isinstance(time_outs, torch.Tensor):
                        time_outs = torch.tensor(time_outs, device=self.device, dtype=torch.bool)
                    else:
                        time_outs = time_outs.to(self.device).bool()
                    terminated = (dones.bool() & ~time_outs).view(-1, 1)
                    truncated = time_outs.view(-1, 1)

                    # BFM trajectory-buffer style: write ALL envs every step.
                    # The ``truncated`` column marks episode boundaries so the
                    # sampler never draws sub-sequences across resets.
                    # ``prev_terminated`` / ``prev_truncated`` are from the
                    # PRECEDING step (matching BFM's layout where each row
                    # stores the done flags that led to this obs).
                    batch = {
                        "observation": obs_dict,
                        "action": actions,
                        "z": z_context,
                        "terminated": prev_terminated,
                        "truncated": prev_truncated,
                        "aux_rewards": aux_rewards_dict,
                    }
                    # Anchored variant: attach the PRE-step world pose captured
                    # above (matches obs_dict's instant — not the post-step pose).
                    if prestep_extras is not None:
                        batch["extras"] = prestep_extras
                    # Safety net: periodic NaN check.
                    if (self.tot_timesteps // self.env.num_envs) % 500 == 0:
                        for k, v in obs_dict.items():
                            if torch.isnan(v).any() or torch.isinf(v).any():
                                print(f"[WARN env_steps={self.tot_timesteps}] "
                                      f"obs[{k}] has nan={int(torch.isnan(v).sum())} "
                                      f"inf={int(torch.isinf(v).sum())}", flush=True)
                    self.replay_buffer.extend(batch)

                    # Update prev flags for the NEXT step's write.
                    # BFM stores terminated/truncated separately. The buffer's
                    # episode segmenter uses truncated only (end_key="truncated"),
                    # but we store the full done signal as truncated so that
                    # both hard-termination and timeout mark episode boundaries
                    # for the sampler. In BFM-Zero production all terminations
                    # are disabled so terminated≡False and truncated≡done.
                    prev_terminated = terminated.clone()
                    prev_truncated = dones.bool().view(-1, 1)

                    # Book-keeping
                    cur_reward_sum += rewards
                    cur_ep_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_ep_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_ep_length[new_ids] = 0

                    # Update per-env step_count (reset on done).
                    step_count = step_count + 1
                    step_count = torch.where(dones.bool(), torch.zeros_like(step_count), step_count)

                    # Resample per-env exploration std for envs that just reset,
                    # so each new episode draws a fresh std in [min, max].
                    if self._use_explore_std_grad and self._explore_std is not None:
                        _dmask = dones.bool()
                        if bool(_dmask.any()):
                            n_d = int(_dmask.sum().item())
                            self._explore_std[_dmask] = self._sample_explore_std(n_d)

                    # Anchored variant: (re)set the per-episode SE(2) anchor A
                    # for envs that just reset, to their fresh spawn pose, so
                    # ``anchored_pose`` (A^-1 g_t) starts near origin. The
                    # displaced rollout anchor (A_init·A_anchor) for tracking
                    # envs is applied just below, after maybe_update_rollout_
                    # context samples A_anchor. The runner-owned FIXED canon
                    # origin ``_canon_*`` is updated to spawn here too (the ONLY
                    # place it moves) so the stored replay frame re-pins at each
                    # episode boundary but stays put across mid-episode resamples.
                    if (self._store_world_pose and _robot is not None
                            and hasattr(self.env_unwrapped, "set_anchor")):
                        done_ids = dones.bool().nonzero(as_tuple=False).squeeze(-1)
                        if done_ids.numel() > 0:
                            rq = _robot.data.root_quat_w.to(self.device)
                            w_, x_, y_, z_ = rq[:, 0], rq[:, 1], rq[:, 2], rq[:, 3]
                            yaw = torch.atan2(2 * (w_ * z_ + x_ * y_),
                                              1 - 2 * (y_ * y_ + z_ * z_))
                            _rxy = _robot.data.root_pos_w[:, :2].to(self.device)
                            self.env_unwrapped.set_anchor(_rxy, yaw, env_ids=done_ids)
                            if getattr(self, "_canon_xy", None) is not None:
                                self._canon_xy[done_ids] = _rxy[done_ids]
                                self._canon_yaw[done_ids] = yaw[done_ids]

                    # Update z context for next step.
                    z_context, terrain_reset = self.alg.maybe_update_rollout_context(
                        z=z_context, step_count=step_count, expert_buffer=self.expert_buffer,
                        robot_root_xy=_robot.data.root_pos_w[:, :2].to(self.device) if _robot else None,
                        robot_root_quat=_robot.data.root_quat_w.to(self.device) if _robot else None,
                        terrain_z_fn=_terrain_z_fn,
                    )
                    # Sync per-env global_root_h flag to the env.
                    grh = getattr(self.alg, "_tracking_terrain_variant_root_h", None)
                    if grh is not None and hasattr(self.env_unwrapped, "_use_terrain_variant_root_h"):
                        self.env_unwrapped._use_terrain_variant_root_h = grh

                    if terrain_reset is not None and hasattr(self.env_unwrapped, "_reset_idx"):
                        env_ids = terrain_reset["env_ids"]
                        # Run the full reset path first so action/observation
                        # histories, episode counters and MDP-owned state cannot
                        # leak across the tracking teleport. Apply the exact RSI
                        # state afterward and mark the next replay row as a
                        # boundary, including envs that also ended naturally.
                        self.env_unwrapped._reset_idx(env_ids)
                        self._terrain_rsi_from_tracking(
                            env_ids,
                            terrain_reset["motion_ids"],
                            terrain_reset["starts"],
                        )
                        step_count[env_ids] = 0
                        dones[env_ids] = 1
                        prev_terminated[env_ids] = False
                        prev_truncated[env_ids] = True
                        fresh_obs, fresh_extras = self.env.get_observations()
                        new_obs = self._obs_to_device(fresh_obs, fresh_extras)
                        if _robot is not None:
                            self.alg.update_tracking_pose_after_reset(
                                env_ids,
                                _robot.data.root_pos_w[:, :2].to(self.device),
                                _robot.data.root_quat_w.to(self.device),
                            )

                    # Two-frame anchor: a tracking resample sampled A_anchor (the
                    # init-local offset). Compute the DISPLACED sim-space env
                    # anchor A_init·A_anchor from the LIVE post-reset robot pose
                    # (terrain RSI above may have just moved tracking envs, so we
                    # MUST read the pose here, after resets settle) and set it as
                    # the env ``anchored_pose`` anchor for tracking envs only —
                    # so the actor obs is in the SAME frame as the displaced
                    # tracking-z. Non-tracking envs keep their spawn anchor; the
                    # runner-owned ``_canon_*`` (stored-pose frame) stays at spawn.
                    self._apply_tracking_sim_anchor(_robot)

                    obs_dict = new_obs

                    # Local drives warmup/update cadence. The raw global counter
                    # drives schedules; the logging counter may cap world size.
                    local_timesteps += self.env.num_envs
                    self._local_timesteps = local_timesteps
                    self.tot_timesteps += self.env.num_envs * self.gpu_world_size
                    self.log_timesteps += self.env.num_envs * self._log_world_size
                    steps_since_last_update += self.env.num_envs

            collection_time = time.time() - start
            start = time.time()

            # ----- updates -----
            loss_dict: Dict[str, float] = {
                "Rollout/tracking_resamples": float(
                    int(getattr(self.alg, "_tracking_resample_count", 0))
                    - tracking_resamples_before
                ),
                "Rollout/tracking_phase": float(
                    int(getattr(self.alg, "_tracking_phase", 0))
                ),
                "Rollout/reset_fraction": (
                    rollout_done_count / max(rollout_transition_count, 1)
                ),
                "Track/early_termination_count": float(
                    tracking_failure_count
                ),
                "Track/bin_success_count": float(
                    tracking_bin_success_count
                ),
                "Track/early_termination_rate": (
                    tracking_failure_count
                    / max(tracking_failure_eligible_steps, 1)
                ),
                "Event/tracking_eval": 0.0,
                "Event/checkpoint": float(it % self.save_interval == 0),
            }
            if tracking_failure_eligible_steps > 0:
                tracking_failure_stats = torch.stack(
                    tracking_failure_stat_chunks
                ).sum(dim=0).detach().cpu().tolist()
                loss_dict["Track/eligible_joint_mae_rad"] = (
                    tracking_failure_stats[0]
                    / tracking_failure_eligible_steps
                )
                loss_dict["Track/eligible_root_height_error_m"] = (
                    tracking_failure_stats[1]
                    / tracking_failure_eligible_steps
                )
                failure_ema = getattr(
                    self.expert_buffer, "_tracking_failure_ema", None
                )
                if failure_ema is not None:
                    failure_ema_stats = torch.stack(
                        (failure_ema.mean(), failure_ema.max())
                    ).detach().cpu().tolist()
                    loss_dict["Track/failure_priority_ema_mean"] = (
                        failure_ema_stats[0]
                    )
                    loss_dict["Track/failure_priority_ema_max"] = (
                        failure_ema_stats[1]
                    )
                bin_success_ema = getattr(
                    self.expert_buffer,
                    "_tracking_bin_success_ema",
                    None,
                )
                if (
                    bin_success_ema is not None
                    and bin_success_ema.numel() > 0
                ):
                    bin_stats = torch.stack(
                        (
                            bin_success_ema.mean(),
                            bin_success_ema.min(),
                            bin_success_ema.max(),
                        )
                    ).detach().cpu().tolist()
                    loss_dict["Track/bin_success_ema_mean"] = bin_stats[0]
                    loss_dict["Track/bin_success_ema_min"] = bin_stats[1]
                    loss_dict["Track/bin_success_ema_max"] = bin_stats[2]
            # Hold off updates until _delay_updates_until (== num_seed_steps on
            # a fresh run; larger on a no-replay resume so the resumed policy
            # refills the empty buffer on-policy first). Note the random-action
            # rollout branch above is gated separately by num_seed_steps, so the
            # extended pre-update collection uses the trained policy, not random.
            warmup_flag = local_timesteps < self._delay_updates_until
            if (
                len(self.replay_buffer) > 0
                and not warmup_flag
                and steps_since_last_update >= self.update_agent_every
            ):
                steps_since_last_update = 0
                self.policy.train()
                batch_size = int(self.alg.cfg.batch_size)
                train_chunks = self.replay_buffer.sample_chunks(
                    batch_size, self.num_agent_updates, target_device=self.device,
                )
                expert_mean_widths = None
                if self.alg._disc_positive_window > 0:
                    expert_mean_widths = torch.cat([
                        self.alg._sample_expert_T(
                            self.alg._disc_num_sequences,
                            int(self.policy.seq_length),
                        )
                        for _ in range(self.num_agent_updates)
                    ])
                expert_chunks = self.expert_buffer.sample_chunks(
                    self.alg._disc_batch_size,
                    self.num_agent_updates,
                    target_device=self.device,
                    mean_widths=expert_mean_widths,
                )
                replay_dict = {
                    "train": _PrefetchedSampler(train_chunks),
                    "expert_slicer": _PrefetchedSampler(expert_chunks),
                }
                for _ in range(self.num_agent_updates):
                    metrics = self.alg.update(replay_dict, step=int(self.tot_timesteps))
                    # Keep detached scalar views and reduce them in one packed
                    # operation after all updates. The old path allocated a
                    # zeros_like default and launched an out-of-place GPU add
                    # for every metric on every update.
                    for k, v in metrics.items():
                        total_metrics.setdefault(k, []).append(v.detach())
                    for k in (
                        "fb_offdiag",
                        "fb_offdiag_row_max",
                        "fb_offdiag_col_max",
                        "fb_innovation_align_loss",
                        "grad_norm/forward_map",
                        "grad_norm/backward_map",
                    ):
                        if k in metrics:
                            max_metrics.setdefault(k, []).append(
                                metrics[k].detach()
                            )

                # Single running-stat sync per iter (was per-update = 16×).
                if self.is_distributed:
                    self.alg._sync_running_stats()

                metric_keys = sorted(total_metrics)
                metric_values = torch.stack(
                    [
                        torch.stack([
                            value.float().mean()
                            for value in total_metrics[k]
                        ]).mean()
                        for k in metric_keys
                    ]
                ).to(self.device)
                if self.is_distributed:
                    torch.distributed.all_reduce(
                        metric_values, op=torch.distributed.ReduceOp.SUM
                    )
                    metric_values.div_(self.gpu_world_size)
                metric_values_host = metric_values.detach().cpu().tolist()
                for k, value in zip(metric_keys, metric_values_host):
                    loss_dict[k] = float(value)

                if max_metrics:
                    max_keys = sorted(max_metrics)
                    max_values = torch.stack([
                        torch.stack([
                            value.float().mean()
                            for value in max_metrics[k]
                        ]).max()
                        for k in max_keys
                    ]).to(self.device)
                    if self.is_distributed:
                        torch.distributed.all_reduce(
                            max_values, op=torch.distributed.ReduceOp.MAX
                        )
                    max_values_host = max_values.detach().cpu().tolist()
                    for k, value in zip(max_keys, max_values_host):
                        loss_dict[f"Spike/{k}_update_rank_max"] = float(value)
                total_metrics = {}
                max_metrics = {}

            # Per-iteration global-tracking deviation (tracking envs only).
            if _trk_count > 0:
                loss_dict["Track/global_xy_dev_m"] = _trk_xy_sum / _trk_count
                loss_dict["Track/global_yaw_dev_deg"] = (
                    _trk_yaw_sum / _trk_count) * 180.0 / math.pi

            # ----- tracking eval (BFM-style, prioritization feedback) -----
            # Fires every ``eval_every_steps`` env-steps once warmup is done
            # (NO initial eval — we want the first eval to happen only after
            # a meaningful number of updates have run). If it fires, we
            # snapshot the env, roll out each motion tracking for
            # ``eval_rollout_length`` env-steps, compute per-motion MPJPE,
            # update the expert buffer's sampling priorities, and restore.
            if (
                not warmup_flag
                and self.eval_every_steps > 0
                and (local_timesteps - self._last_eval_step) >= self.eval_every_steps
            ):
                self._last_eval_step = local_timesteps
                eval_metrics = self._run_tracking_eval()
                if eval_metrics is not None:
                    loss_dict["Event/tracking_eval"] = 1.0
                    # Treat eval as a rollout boundary even though the env
                    # snapshot now restores hidden state. This guarantees that
                    # an overlooked simulator/sensor cache can never create a
                    # pre-eval -> post-eval replay transition.
                    # The previous flags were produced inside inference_mode,
                    # so they are inference tensors and cannot be mutated here.
                    if bool(self.alg_cfg.get("replay_mark_eval_boundary", True)):
                        prev_terminated = torch.zeros(
                            (self.env.num_envs, 1),
                            dtype=torch.bool,
                            device=self.device,
                        )
                        prev_truncated = torch.ones(
                            (self.env.num_envs, 1),
                            dtype=torch.bool,
                            device=self.device,
                        )
                    for k, v in eval_metrics.items():
                        loss_dict[k] = v

            learn_time = time.time() - start
            self.current_learning_iteration = it
            self.tot_time += collection_time + learn_time

            # ----- log (head rank only) -----
            if self._is_head and self.log_dir is not None and self.writer is not None:
                self._log(it, tot_iter, collection_time, learn_time, loss_dict, rewbuffer, lenbuffer)
                if it % self.save_interval == 0:
                    if len(rewbuffer) > 0:
                        current_reward = statistics.mean(rewbuffer)
                        if current_reward > best_reward:
                            best_reward = current_reward
                            # model_best: policy only — resumed checkpoints
                            # come from the iter saves below, not from
                            # model_best.
                            self.save(os.path.join(self.log_dir, "model_best.pt"))
                    # Save the light (policy + optim + meta) checkpoint
                    # every save_interval. The heavy replay sibling is
                    # saved every ``save_replay_every_n`` light-save
                    # intervals — large enough to avoid bloating disk
                    # (~5 GB per replay), small enough to allow warm-start.
                    save_iter_path = os.path.join(self.log_dir, f"model_{it}.pt")
                    n_between = int(self.alg_cfg.get("save_replay_every_n", 10))
                    save_replay = n_between > 0 and ((it // self.save_interval) % n_between == 0)
                    self.save(save_iter_path, save_replay=save_replay)
                    # Mirror the light checkpoint to S3 in the background
                    # (rolling single object). No-op if s3_ckpt_uri unset.
                    self._mirror_checkpoint_to_s3(save_iter_path)
                    # Ring-buffer pruning: keep only the last
                    # ``keep_last_n_checkpoints`` model_<iter>.pt files (and
                    # their optional .replay.pt siblings). Each light ckpt
                    # is ~3-7 GB and each replay ckpt adds ~20-40 GB, so
                    # without pruning a long run fills the node's boot
                    # disk — which kills the ray agent and puts the sky
                    # cluster into INIT. ``model_best.pt`` is never pruned.
                    keep_n = int(self.alg_cfg.get("keep_last_n_checkpoints", 10))
                    if keep_n > 0:
                        self._prune_old_checkpoints(keep_n)

            if self._is_head and it == start_iter and self.log_dir is not None:
                import rsl_rl
                store_code_state(self.log_dir, [rsl_rl.__file__])

        self._local_timesteps = local_timesteps
        if self._is_head and self.log_dir is not None:
            # Final save: always include replay so the next resume can
            # warm-start without rebuilding the buffer from scratch.
            final_path = os.path.join(
                self.log_dir, f"model_{self.current_learning_iteration}.pt")
            self.save(final_path, save_replay=True)
            # Final S3 mirror. Wait for any in-flight upload to finish first so
            # this last checkpoint isn't skipped, then BLOCK on it (bounded) so
            # the process doesn't exit mid-upload.
            if self.s3_ckpt_uri:
                if self._s3_thread is not None and self._s3_thread.is_alive():
                    self._s3_thread.join(timeout=600)
                self._mirror_checkpoint_to_s3(final_path)
                if self._s3_thread is not None:
                    self._s3_thread.join(timeout=600)

    # --- B-spectrum diagnostic ------------------------------------------- #

    @torch.inference_mode()
    def _backward_spectrum_metrics(self, num_samples: int = 10_000) -> Dict[str, float] | None:
        """Eigen-spectrum of the backward-feature gram E[B(s)^T B(s)].

        Draws ~``num_samples`` i.i.d. obs from the replay buffer, encodes them
        through the (sphere-projected) backward map ``B(s) -> R^{z_dim}``, and
        forms the second-moment matrix ``M = (1/N) B^T B`` (z_dim x z_dim).
        Its eigenvalues describe how B spreads its representation across the
        z_dim directions — a collapsed B concentrates mass in a few eigenvalues
        (low effective rank), a well-conditioned one spreads it out.

        Returns ``Eval/B_*`` scalars (effective rank + spectrum summary), or
        ``None`` if the buffer is empty / B is unavailable. Under DDP each rank
        samples its own local replay; the head rank's numbers are logged.
        """
        if len(self.replay_buffer) == 0:
            return None
        policy = self.policy
        if not hasattr(policy, "backward_map"):
            return None

        # Gather ~num_samples B(s) vectors in chunks (cap chunk so a single
        # backward forward-pass stays well within memory).
        chunk = 4096
        feats: list[torch.Tensor] = []
        collected = 0
        was_training = policy.training
        policy.eval()
        try:
            while collected < num_samples:
                bs = min(chunk, num_samples - collected)
                try:
                    batch = self.replay_buffer.sample_flat(bs)
                except RuntimeError:
                    break
                z = policy.backward_map(batch["observation"])  # [bs, z_dim]
                feats.append(z.float())
                collected += int(z.shape[0])
        finally:
            if was_training:
                policy.train()
        if not feats:
            return None

        B = torch.cat(feats, dim=0)                      # [N, z_dim]
        N, z_dim = B.shape
        gram = (B.transpose(0, 1) @ B) / float(N)        # [z_dim, z_dim] = E[B^T B]
        # Symmetrize for numerical safety, then real symmetric eigvals (ascending).
        gram = 0.5 * (gram + gram.transpose(0, 1))
        eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)   # ascending
        eig_desc = torch.flip(eigvals, dims=(0,))              # descending

        total = eig_desc.sum().clamp_min(1e-12)
        # Effective rank (a.k.a. "erank"): exp of the spectral entropy of the
        # normalized eigenvalue distribution. Ranges in [1, z_dim]; equals
        # z_dim for a flat (white) spectrum, ->1 for full collapse.
        p = (eig_desc / total).clamp_min(1e-12)
        spectral_entropy = -(p * p.log()).sum()
        effective_rank = float(spectral_entropy.exp().item())
        # Participation ratio: (sum λ)^2 / sum(λ^2) — a second, threshold-free
        # rank proxy (heavier-tailed than erank).
        participation_ratio = float((total * total / (eig_desc * eig_desc).sum().clamp_min(1e-12)).item())
        # Numerical rank: # eigenvalues above 1% of the largest.
        lam_max = float(eig_desc[0].item())
        thresh = 0.01 * lam_max
        numerical_rank = float((eig_desc > thresh).sum().item())
        condition_number = float((lam_max / eig_desc[eig_desc > 1e-12].min().clamp_min(1e-12)).item())

        out = {
            "Eval/B_effective_rank": effective_rank,
            "Eval/B_participation_ratio": participation_ratio,
            "Eval/B_numerical_rank": numerical_rank,
            "Eval/B_eig_max": lam_max,
            "Eval/B_eig_min": float(eig_desc[-1].item()),
            "Eval/B_eig_mean": float((total / z_dim).item()),
            "Eval/B_condition_number": condition_number,
            "Eval/B_spectrum_samples": float(N),
        }
        # Spectrum shape: log the cumulative energy captured by the top-k
        # eigenvalues (k as a fraction of z_dim) — a scalar-friendly summary of
        # the full spectrum that wandb can curve over training.
        cum = torch.cumsum(eig_desc, dim=0) / total
        for frac in (0.01, 0.05, 0.10, 0.25, 0.50):
            k = max(1, int(round(frac * z_dim)))
            out[f"Eval/B_energy_top{int(frac * 100):02d}pct"] = float(cum[min(k, z_dim) - 1].item())
        return out

    # --- tracking eval --------------------------------------------------- #

    @torch.inference_mode()
    def _run_tracking_eval(self, update_priorities: bool | None = None) -> Dict[str, float] | None:
        """BFM-style tracking eval over every motion in the expert buffer.

        Args:
            update_priorities: Override for self.eval_update_priorities.
                Set to False for the initial (pre-training) baseline eval
                so the untrained policy doesn't poison the expert-buffer
                sampling weights.

        Pipeline:
          1. Snapshot the env.
          2. Assign motions to envs (cycle through all motions). For each env,
             compute the per-motion z context via B(next_obs).mean(seq_window)
             → project_z.
          3. Reset each env to its motion's frame 0 (RSI).
          4. Roll out the policy for ``eval_rollout_length`` steps using
             ``policy.act(..., mean=True)``.
          5. Compute per-motion MPJPE (joint_pos L2 error, in mm).
          6. (optional) Update expert buffer's sampling weights via MPJPE.
          7. Restore env.

        Returns ``Eval/*`` scalars, or ``None`` if the expert buffer lacks
        RSI fields.
        """
        do_update_priors = self.eval_update_priorities if update_priorities is None else bool(update_priorities)
        num_envs = self.env.num_envs
        num_motions = self.expert_buffer.num_unique_motions
        env_u = self.env_unwrapped
        local_ready = bool(
            getattr(self.expert_buffer, "supports_reset_states", False)
            and num_motions > 0
            and hasattr(env_u, "snapshot_state")
            and hasattr(env_u, "restore_state")
        )
        if (
            self.is_distributed
            and bool(self.alg_cfg.get("distributed_expert", False))
            and torch.distributed.is_initialized()
        ):
            ready = torch.tensor(
                int(local_ready), device=self.device, dtype=torch.int32
            )
            torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
            local_ready = bool(ready.item())
        if not local_ready:
            return None

        snap = env_u.snapshot_state()
        try:
            # --- assign a motion id to each env (cycle, shuffled) ---
            # BFM shuffles env indices before assigning motions so that each
            # motion gets evaluated with multiple DR instances (mass/friction
            # are randomized per-env at startup and stay fixed for the whole
            # run). Without shuffling, motion m is always evaluated with
            # env m's specific DR settings — biasing per-motion MPJPE by the
            # DR instance. We use a deterministic shuffle (seed=0) so eval
            # results are reproducible across runs.
            g = torch.Generator(device="cpu")
            g.manual_seed(0)
            perm_cpu = torch.randperm(num_envs, generator=g)
            shuffled_env_idxs = perm_cpu.to(self.device)
            motion_of_env = torch.zeros(num_envs, device=self.device, dtype=torch.long)
            motion_of_env[shuffled_env_idxs] = (
                torch.arange(num_envs, device=self.device) % num_motions
            )

            # --- per-motion windows of length L = eval_rollout_length+1 ---
            L = int(self.eval_rollout_length) + 1
            seq_length = int(self.policy.seq_length)
            action_dim = int(self.action_dim)
            num_joints = int(self.expert_buffer.num_joints)

            # Pre-encode a time-major CPU schedule for the motions that are
            # actually assigned to an environment. Keeping motion (rather than
            # environment) rows avoids duplicating z for the multiple DR
            # replicas of each motion. At rollout, one contiguous transfer plus
            # one GPU index_select replaces the old per-motion mask/transfer
            # loop at every timestep.
            num_assigned_motions = min(num_motions, num_envs)
            z_motion_schedule = torch.empty(
                L - 1,
                num_assigned_motions,
                self.policy.z_dim,
                dtype=torch.float32,
                device="cpu",
            )
            for m in range(num_assigned_motions):
                win = self.expert_buffer.get_motion_window(m, num_frames=L)
                next_obs_dict = {
                    "state": win["state"][1:].to(self.device, non_blocking=True),
                    "privileged_state": win["privileged_state"][1:].to(self.device, non_blocking=True),
                    "last_action": win["last_action"][1:].to(self.device, non_blocking=True),
                    "history_actor": win["history_actor"][1:].to(self.device, non_blocking=True),
                }
                # Anchored variant: B also reads ``anchored_pose`` (A^-1 g). For
                # tracking eval the goal is self-anchored (anchor = each frame's
                # own pose), i.e. zero displacement -> [0, 0, cos0, sin0]=[0,0,1,0].
                if "anchored_pose" in tuple(self.policy_cfg.get("backward_input_keys", ())):
                    n = next_obs_dict["state"].shape[0]
                    ap = torch.zeros(n, 4, device=self.device)
                    ap[:, 2] = 1.0  # cos(theta=0)
                    next_obs_dict["anchored_pose"] = ap
                z = self.policy.backward_map(next_obs_dict)   # [L-1, z_dim]
                # Match UFO's eval z-encoding exactly: one next-reference frame,
                # i.e. no temporal averaging is needed here.
                z = self.policy.project_z(z)
                if z.shape[0] == 0:
                    z_motion_schedule[:, m].zero_()
                    continue
                z_cpu = z.to(device="cpu", dtype=torch.float32)
                n = min(int(z_cpu.shape[0]), L - 1)
                z_motion_schedule[:n, m].copy_(z_cpu[:n])
                # Match the old min(t, motion_length - 1) behavior for motions
                # shorter than the fixed evaluation rollout.
                if n < L - 1:
                    z_motion_schedule[n:, m].copy_(z_cpu[n - 1])

            # --- reset envs to each motion's frame-0 state ---
            # Build aligned per-env buffers.
            jp0 = torch.zeros(num_envs, num_joints, device=self.device)
            jv0 = torch.zeros_like(jp0)
            rp0 = torch.zeros(num_envs, 3, device=self.device)
            rq0 = torch.zeros(num_envs, 4, device=self.device); rq0[:, 0] = 1.0
            rv0 = torch.zeros(num_envs, 3, device=self.device)
            rav0 = torch.zeros(num_envs, 3, device=self.device)
            for m in range(num_motions):
                win = self.expert_buffer.get_motion_window(m, num_frames=1)
                if win["num_frames"] < 1:
                    continue
                mask = motion_of_env == m
                if not mask.any():
                    continue
                jp0[mask] = win["joint_pos"][0:1].to(self.device)
                jv0[mask] = win["joint_vel"][0:1].to(self.device)
                rp0[mask] = win["root_pos"][0:1].to(self.device)
                rq0[mask] = win["root_quat"][0:1].to(self.device)
                rv0[mask] = win["root_lin_vel"][0:1].to(self.device)
                rav0[mask] = win["root_ang_vel"][0:1].to(self.device)

            # Set env into eval mode (skip reset-source counters)
            if hasattr(env_u, "_eval_mode"):
                env_u._eval_mode = True
            env_ids_all = env_u._ALL_INDICES
            # BFM's eval calls env.reset(target_states=...) which triggers the
            # full reset logic (clears action_history, episode_length_buf, obs
            # history buffers, MDP term private state) AND writes the target
            # pose/joints. We reproduce this: first call _reset_idx on all
            # envs (clears all training-time stale state), then override the
            # joint/root state to the motion's frame-0. Without this, the
            # eval rollout would use stale history_actor / action_history /
            # last_actions from the paused training rollout — corrupting
            # the policy's obs and producing biased MPJPE.
            env_u._reset_idx(env_ids_all)
            # Write the motion-aligned initial state directly to the sim
            # (overrides whatever _reset_idx wrote via the normal reset path).
            joint_pos_full = torch.zeros(num_envs, env_u.robot.data.joint_pos.shape[1], device=self.device)
            joint_vel_full = torch.zeros_like(joint_pos_full)
            joint_order_t = torch.as_tensor(env_u.joint_order, device=self.device, dtype=torch.long)
            joint_pos_full[:, joint_order_t] = jp0
            joint_vel_full[:, joint_order_t] = jv0
            env_u.robot.write_joint_position_to_sim(joint_pos_full, env_ids=env_ids_all)
            env_u.robot.write_joint_velocity_to_sim(joint_vel_full, env_ids=env_ids_all)
            # RSI z is relative to env origin.
            rp0_abs = rp0.clone()
            rp0_abs[:, :2] = rp0_abs[:, :2] + env_u.scene.env_origins[:, :2]
            env_u.robot.write_root_pose_to_sim(
                torch.cat([rp0_abs, rq0], dim=-1), env_ids=env_ids_all
            )
            env_u.robot.write_root_velocity_to_sim(
                torch.cat([rv0, rav0], dim=-1), env_ids=env_ids_all
            )
            env_u.scene.write_data_to_sim()
            env_u.sim.forward()
            # Refresh last_* joint buffers to the NEW state (they were set
            # to the pre-write sim state by _reset_idx; we need them to
            # reflect the post-write state so the first obs/reward is clean).
            env_u._refresh_sim_tensors(env_ids_all)
            env_u.last_joint_pos[env_ids_all] = env_u.joint_pos[env_ids_all]
            env_u.last_joint_vel[env_ids_all] = env_u.joint_vel[env_ids_all]

            # --- rollout ---
            obs_flat, extras = self.env.get_observations()
            obs_dict = self._obs_to_device(obs_flat, extras)

            # Per-env joint_pos log (for MPJPE) and dof_pos_dev log (for EMD).
            jp_log = torch.zeros(num_envs, L, num_joints, device=self.device)
            # dof_pos_dev = first num_joints dims of the env's `state` obs key
            dpd_log = torch.zeros(num_envs, L, num_joints, device=self.device)
            # env_u.joint_pos is ALREADY in the canonical (cfg.robot_joint_order)
            # order — see base_env.py:298 where it's built as
            # ``robot.data.joint_pos[:, self.joint_order]``. Expert buffer's
            # joint_pos is in the same canonical order. So direct comparison,
            # NO additional reindexing. (Previous code did ``[:, joint_order_t]``
            # which double-indexed and permuted the joints, massively inflating
            # Eval/mpjpe_mm.)
            jp_log[:, 0] = env_u.joint_pos
            dpd_log[:, 0] = obs_dict["state"][:, :num_joints]
            for t in range(1, L):
                # One H2D transfer for all assigned motions, then replicate
                # motion z across their DR environments entirely on the GPU.
                z_motion_t = z_motion_schedule[t - 1].to(
                    self.device, non_blocking=True
                )
                z_batch = z_motion_t.index_select(0, motion_of_env)
                action = self.policy.act(obs_dict, z_batch, mean=True)
                new_obs, _, _, infos = self.env.step(action.to(self.env.device))
                obs_dict = self._obs_to_device(new_obs, infos)
                jp_log[:, t] = env_u.joint_pos
                dpd_log[:, t] = obs_dict["state"][:, :num_joints]

            # --- per-motion metrics ---
            # MPJPE (mm) — informative log-only metric.
            # EMD (optimal-transport distance between rollout and ref dof_pos_dev
            # sequences) — BFM's tracking fitness metric, used for priority feedback.
            import ot as _ot
            import numpy as _np
            mpjpe_per_motion = torch.zeros(num_motions, device=self.device)
            emd_per_motion = torch.zeros(num_motions, device=self.device)
            count_per_motion = torch.zeros(num_motions, device=self.device)
            emd_jobs = []

            def _solve_uniform_emd(cost):
                a = _np.ones(cost.shape[0], dtype=_np.float64) / cost.shape[0]
                b = _np.ones(cost.shape[1], dtype=_np.float64) / cost.shape[1]
                try:
                    return float(
                        _ot.emd2(
                            a,
                            b,
                            cost,
                            numItermax=100_000,
                            numThreads=1,
                        )
                    )
                except Exception:
                    return float("nan")

            # POT's exact network-simplex solver is CPU-only. Pipeline GPU
            # cdist preparation with a bounded pool of independent EMD solves;
            # each solve stays single-threaded to avoid nested oversubscription
            # when many DDP ranks share one host.
            num_emd_workers = min(self.eval_emd_workers, num_assigned_motions)
            with ThreadPoolExecutor(
                max_workers=num_emd_workers,
                thread_name_prefix="bfm-eval-emd",
            ) as emd_pool:
                for m in range(num_motions):
                    win = self.expert_buffer.get_motion_window(m, num_frames=L)
                    T_m = int(win["num_frames"])
                    if T_m < 2:
                        continue
                    target_jp = win["joint_pos"][:T_m].to(self.device)     # [T_m, J]
                    target_state = win["state"][:T_m, :num_joints].to(self.device)  # [T_m, J]
                    mask = motion_of_env == m
                    if not mask.any():
                        continue
                    # MPJPE: per-env L2 error in mm (averaged over joints then over time).
                    env_jp = jp_log[mask, :T_m]                            # [N_env, T_m, J]
                    err = torch.norm(env_jp - target_jp.unsqueeze(0), dim=-1).mean(dim=-1) * 1000.0
                    mpjpe_per_motion[m] = err.mean()
                    # EMD: optimal transport on the rollout's dof_pos_dev sequence vs the
                    # motion's dof_pos_dev sequence. BFM uses `state[:, :QVEL_IDX]`, i.e.
                    # the dof_pos block only. Take the first env assigned to this motion
                    # (BFM averages across per-motion envs; for a single-rep eval picking
                    # the first is equivalent).
                    env_idxs = mask.nonzero(as_tuple=False).view(-1)
                    first_env = int(env_idxs[0].item())
                    agent_seq = dpd_log[first_env, :T_m].detach()          # [T_m, J]
                    ref_seq = target_state                                 # [T_m, J]
                    # Prepare pairwise L2 cost on GPU, then submit the exact
                    # uniform-mass OT solve immediately so CPU and GPU work overlap.
                    cost = torch.cdist(agent_seq, ref_seq, p=2).detach().cpu().numpy()
                    emd_jobs.append((m, emd_pool.submit(_solve_uniform_emd, cost)))
                    count_per_motion[m] = 1.0

                for m, future in emd_jobs:
                    emd_per_motion[m] = future.result()

            valid = count_per_motion > 0
            if not bool(valid.any().item()):
                return None
            mean_mpjpe = float(mpjpe_per_motion[valid].mean().item())
            # Scrub any NaN/inf from the EMD tensor before aggregation.
            emd_clean = torch.where(torch.isfinite(emd_per_motion), emd_per_motion,
                                    torch.zeros_like(emd_per_motion))
            mean_emd = float(emd_clean[valid].mean().item())
            # Success rate: fraction of motions with MPJPE < 500 mm (PHC-style).
            success = float((mpjpe_per_motion[valid] < 500.0).float().mean().item())

            # --- prioritization feedback (BFM: uses EMD, not MPJPE) ---
            # BFM train.py:360-386: priorities = EMD per motion, clamped to
            # [prioritization_min_val, prioritization_max_val], scaled by
            # prioritization_scale, then mapped via mode (bin / exp / lin).
            # Production: min=0.5, max=2.0, scale=2.0, mode='exp'.
            if do_update_priors:
                w = emd_clean.clone()
                w[~valid] = w[valid].mean() if valid.any() else 1.0
                w_clamped = w.clamp(self.eval_priority_min, self.eval_priority_max)
                w_scaled = w_clamped * self.eval_priority_scale
                if self.eval_priority_mode == "exp":
                    # BFM uses `priorities = 2 ** priorities`.
                    w_final = torch.pow(2.0, w_scaled)
                elif self.eval_priority_mode == "bin":
                    # BFM bin-mode: floor -> per-bin uniform reweight.
                    bins = torch.floor(w_scaled)
                    w_final = torch.ones_like(w_scaled)
                    for b_val in torch.unique(bins).tolist():
                        m_mask = bins == b_val
                        n_in_bin = int(m_mask.sum().item())
                        if n_in_bin > 0:
                            w_final[m_mask] = 1.0 / n_in_bin
                else:  # "lin"
                    w_final = w_scaled
                # update_priorities normalizes internally to sum=1.
                self.expert_buffer.update_priorities(w_final)

            n_valid = int(valid.sum().item())
            out = {
                "Eval/mpjpe_mm": mean_mpjpe,
                "Eval/emd": mean_emd,
                "Eval/tracking_success": success,
                "Eval/num_motions": float(n_valid),
            }

            # Distributed expert: average over ranks weighted by each
            # rank's local valid-motion count, so the reported number is
            # over the GLOBAL (union-of-shards) motion set.
            if (
                self.is_distributed
                and bool(self.alg_cfg.get("distributed_expert", False))
                and torch.distributed.is_initialized()
            ):
                dev = torch.device(self.device)
                metric_names = ["Eval/mpjpe_mm", "Eval/emd", "Eval/tracking_success"]
                # Pack [mpjpe * n, emd * n, success * n, n] as sums to
                # all-reduce in a single call, then recover weighted
                # means by dividing by the total count.
                packed = torch.tensor(
                    [out[k] * n_valid for k in metric_names] + [float(n_valid)],
                    dtype=torch.float64, device=dev,
                )
                torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
                total_n = packed[-1].clamp_min(1.0)
                for i, k in enumerate(metric_names):
                    out[k] = float((packed[i] / total_n).item())
                out["Eval/num_motions"] = float(packed[-1].item())

            # Backward-feature eigen-spectrum (E[B^T B]) — local replay, head
            # rank's numbers are the ones logged. Not all-reduced: it's a
            # rank-local diagnostic of B's representation spread.
            try:
                spec = self._backward_spectrum_metrics()
                if spec is not None:
                    out.update(spec)
            except Exception as e:  # never let a diagnostic kill eval
                if self._is_head:
                    print(f"[FBCprRunner] B-spectrum diagnostic failed: {e}", flush=True)
            return out
        finally:
            if hasattr(env_u, "_eval_mode"):
                env_u._eval_mode = False
            env_u.restore_state(snap)
            # Reclaim the transient GPU allocations from the per-motion eval
            # rollout. torch.compile recompiles per motion-window shape and
            # leaves reserved-but-unallocated fragmentation; the eval scratch
            # (jp/dpd logs, z_batch) is freed as this frame unwinds, so an
            # empty_cache here returns those blocks to the allocator before the
            # next training update instead of fragmenting its headroom.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- utilities ------------------------------------------------------- #

    @torch.inference_mode()
    def _terrain_rsi_from_tracking(
        self,
        env_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        starts: torch.Tensor,
        *,
        align_to_env_origins: bool = False,
    ) -> None:
        """Reset specific envs to the tracking context's motion/frame.

        Uses ``get_reset_states_at`` so the robot is placed at the exact
        same motion frame that the tracking z was encoded from.
        """
        eu = self.env_unwrapped
        states = self.expert_buffer.get_reset_states_at(
            motion_ids.to(self.expert_buffer.device),
            starts.to(self.expert_buffer.device),
        )
        n = env_ids.shape[0]
        dev = self.device
        jp_canon = states["joint_pos"].to(dev)
        jv_canon = states["joint_vel"].to(dev)
        rp = states["root_pos"].to(dev)
        rq = states["root_quat"].to(dev)
        rlv = states["root_lin_vel"].to(dev)
        rav = states["root_ang_vel"].to(dev)
        joint_order_t = torch.as_tensor(eu.joint_order, device=dev, dtype=torch.long)
        jp_usd = torch.zeros(n, eu.robot.data.joint_pos.shape[1], device=dev)
        jv_usd = torch.zeros_like(jp_usd)
        jp_usd[:, joint_order_t] = jp_canon
        jv_usd[:, joint_order_t] = jv_canon
        # Normal flat vector envs live in separate world tiles. Dataset root XY
        # is motion-local, so a synthetic tracking reset must retain the exact
        # reference pose/velocity while placing its root in that env's tile.
        if align_to_env_origins and hasattr(eu.scene, "env_origins"):
            rp = rp.clone()
            rp[:, :2] = eu.scene.env_origins[env_ids, :2].to(dev)
            rp[:, 2] = rp[:, 2] + float(
                getattr(eu.cfg, "rsi_z_margin", 0.0)
            )
        # No-anchor / origin-spawn: force the RSI root to world origin (XY=0)
        # facing +x, keeping the motion's joint pose + z. This matches the env's
        # spawn_at_origin reset so a mid-episode tracking resample restarts the
        # new motion's z window from the SAME canonical origin frame the z was
        # encoded in (anchor pinned at origin). Velocities zeroed (start at rest).
        if bool(getattr(eu.cfg, "spawn_at_origin", False)):
            rp = rp.clone(); rp[:, :2] = 0.0
            rq = torch.zeros_like(rq); rq[:, 0] = 1.0
            rlv = torch.zeros_like(rlv); rav = torch.zeros_like(rav)
        eu.robot.write_joint_position_to_sim(jp_usd, env_ids=env_ids)
        eu.robot.write_joint_velocity_to_sim(jv_usd, env_ids=env_ids)
        eu.robot.write_root_pose_to_sim(
            torch.cat([rp, rq], dim=-1), env_ids=env_ids)
        eu.robot.write_root_velocity_to_sim(
            torch.cat([rlv, rav], dim=-1), env_ids=env_ids)
        eu.scene.write_data_to_sim()
        eu.sim.forward()
        if hasattr(eu, "_refresh_sim_tensors"):
            eu._refresh_sim_tensors(env_ids)
        if hasattr(eu, "last_joint_pos"):
            eu.last_joint_pos[env_ids] = eu.joint_pos[env_ids]
        if hasattr(eu, "last_joint_vel"):
            eu.last_joint_vel[env_ids] = eu.joint_vel[env_ids]

    @staticmethod
    def _clone_observation_history(history):
        if isinstance(history, torch.Tensor):
            return history.clone()
        if isinstance(history, dict):
            return {
                key: value.clone() for key, value in history.items()
            }
        if isinstance(history, (tuple, list)):
            return type(history)(value.clone() for value in history)
        return None

    def _snapshot_observation_histories(self) -> list[tuple[Any, Any]]:
        """Snapshot observation rings before an out-of-band subset reset."""
        obs_cfg = getattr(
            self.env_unwrapped, "main_observation_cfg", None
        )
        if obs_cfg is None:
            return []
        snapshots = []
        for term in obs_cfg.observation_terms.values():
            try:
                history = term._get_from_private_buffer("history", None)
            except (AttributeError, KeyError):
                continue
            cloned = self._clone_observation_history(history)
            if cloned is not None:
                snapshots.append((term, cloned))
        return snapshots

    def _restore_observation_histories(
        self,
        snapshots: list[tuple[Any, Any]],
        reset_env_ids: torch.Tensor,
    ) -> None:
        """Undo the extra observation push for non-reset environments."""
        keep = torch.ones(
            self.env.num_envs, dtype=torch.bool, device=self.device
        )
        keep[reset_env_ids] = False

        def _restore(current, saved) -> None:
            if isinstance(current, torch.Tensor):
                current[keep] = saved[keep]
            elif isinstance(current, dict):
                for key in current:
                    current[key][keep] = saved[key][keep]
            elif isinstance(current, (tuple, list)):
                for value, saved_value in zip(current, saved):
                    value[keep] = saved_value[keep]

        for term, saved in snapshots:
            current = term._get_from_private_buffer("history", None)
            if current is None:
                continue
            _restore(current, saved)
            term._set_to_private_buffer("history", current)

    def _obs_to_device(self, obs: Any, extras: dict | None = None) -> Dict[str, torch.Tensor]:
        """Convert the env's flat ``policy`` + ``critic`` tensors into a BFM-agent dict.

        Accepts either:
          * the dict ``{policy, critic, ...}`` returned by
            ``env_unwrapped._get_observations()`` directly, or
          * the flat ``policy`` tensor returned by ``env.get_observations()``
            wrappers, in which case ``extras['observations']['critic']`` must
            hold the critic tensor.
        """
        if isinstance(obs, dict) and ("policy" in obs or "critic" in obs):
            policy_flat = obs.get("policy")
            critic_flat = obs.get("critic", policy_flat)
        elif isinstance(obs, torch.Tensor):
            policy_flat = obs
            if extras is not None and "observations" in extras and "critic" in extras["observations"]:
                critic_flat = extras["observations"]["critic"]
            else:
                critic_flat = policy_flat
        else:
            # Already a BFM-keyed dict.
            return {k: v.to(self.device, non_blocking=True) for k, v in obs.items()}
        return self._flat_to_dict(policy_flat, critic_flat)

    def _extract_aux_rewards(self, infos: dict) -> Dict[str, torch.Tensor]:
        aux = infos.get("aux_rewards", None)
        if aux is None:
            return {}
        out: Dict[str, torch.Tensor] = {}
        for name, vals in aux.items():
            if name.startswith("_"):
                continue
            if not isinstance(vals, torch.Tensor):
                vals = torch.tensor(vals, device=self.device)
            out[name] = vals.to(self.device).view(-1, 1).float()
        return out

    def _log(
        self,
        it: int,
        tot_iter: int,
        collection_time: float,
        learn_time: float,
        loss_dict: Dict[str, float],
        rewbuffer: deque,
        lenbuffer: deque,
    ) -> None:
        env_steps = int(self.log_timesteps)
        for tag, val in loss_dict.items():
            self.writer.add_scalar(tag, val, env_steps)
        if len(rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), env_steps)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), env_steps)
        fps = int(self.env.num_envs * self.num_steps_per_env / max(collection_time + learn_time, 1e-6))

        summary = (
            f"[FBCpr] iter {it}/{tot_iter}  env_steps={env_steps:,}  "
            f"coll={collection_time:.2f}s  learn={learn_time:.2f}s  fps={fps}"
        )
        if len(rewbuffer) > 0:
            summary += f"  mean_rew={statistics.mean(rewbuffer):.3f}"
        # Print the key FB-CPR-Aux losses inline when the update fired.
        loss_keys_to_print = (
            "fb_loss",
            "fb_innovation_align_loss",
            "orth_loss",
            "critic_loss",
            "aux_critic_loss",
            "disc_loss",
            "actor_loss",
            "Q_fb",
            "Q_discriminator",
            "Q_aux",
            "mean_disc_reward",
            "mean_aux_reward",
            "aux_reward_sigma_ema",
            # Manifold attractor
            "ma_loss",
            # Soft FB
            "log_pi_mean",
            "beta_z_mean",
            "entropy_critic_loss",
            "soft_actor_core_loss",
            "q_h_mean",
            "actor_log_std_mean",
            # Eval
            "Eval/mpjpe_mm",
            "Eval/tracking_success",
            # Global-tracking deviation (tracking envs)
            "Track/global_xy_dev_m",
            "Track/global_yaw_dev_deg",
        )
        for k in loss_keys_to_print:
            if k in loss_dict:
                summary += f"  {k}={loss_dict[k]:.3g}"
        print(summary, flush=True)

    # --- checkpoint I/O -------------------------------------------------- #

    def save(
        self,
        path: str,
        infos: Any | None = None,
        *,
        save_replay: bool = False,
    ) -> None:
        """Save a checkpoint.

        By default saves a LIGHT checkpoint (no replay buffer) — fast,
        small, and sufficient for policy evaluation / sim2sim.

        When ``save_replay=True`` ALSO writes a ``<stem>.replay.pt``
        sibling file containing the replay buffer state. Load with
        ``load(..., load_replay=True)`` to resume training from the
        exact buffer state. The replay file is large (~5 GB for the
        production config), so we only write it periodically.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        base = {
            "model": self.policy.state_dict(),
            "optimizers": self.alg.optimizer_dict,
            "algorithm_state": self.alg.training_state_dict,
            "iter": self.current_learning_iteration,
            "tot_timesteps": self.tot_timesteps,
            "log_timesteps": self.log_timesteps,
            "world_size": self.gpu_world_size,
            "local_timesteps": getattr(self, "_local_timesteps", 0),
            "last_eval_step": self._last_eval_step,
            "expert_priority_state": (
                self.expert_buffer.priority_state_dict()
                if (
                    hasattr(self.expert_buffer, "priority_state_dict")
                    and not (
                        self.is_distributed
                        and bool(
                            self.alg_cfg.get("distributed_expert", False)
                        )
                    )
                )
                else None
            ),
            "infos": infos or {},
        }
        torch.save(base, path)
        if save_replay:
            # Separate sibling file keeps the lightweight ckpt small even
            # when we periodically snapshot the replay too. Path convention:
            #   model_<iter>.pt           (light, policy + opt + meta)
            #   model_<iter>.replay.pt    (heavy, replay buffer state)
            replay_path = _replay_sibling_path(path)
            try:
                torch.save({"replay": self.replay_buffer.state_dict()}, replay_path)
            except Exception as e:
                print(f"[FBCprRunner] WARN: failed to save replay to {replay_path}: {e}",
                      flush=True)

    def _prune_old_checkpoints(self, keep_n: int) -> None:
        """Keep only the ``keep_n`` newest ``model_<iter>.pt`` files (and their
        ``.replay.pt`` siblings). Leaves ``model_best.pt`` alone. Called
        right after each save; cheap glob + unlink, so no notable overhead.
        """
        import glob
        import re
        if self.log_dir is None:
            return
        pattern = os.path.join(self.log_dir, "model_*.pt")
        paths = glob.glob(pattern)
        # Skip model_best.pt and any *.replay.pt (siblings handled alongside).
        iter_re = re.compile(r"model_(\d+)\.pt$")
        iter_paths: list[tuple[int, str]] = []
        for p in paths:
            m = iter_re.search(os.path.basename(p))
            if m is None:
                continue
            iter_paths.append((int(m.group(1)), p))
        if len(iter_paths) <= keep_n:
            return
        iter_paths.sort(key=lambda x: x[0])   # oldest first
        to_remove = iter_paths[:-keep_n]
        for _, p in to_remove:
            try:
                os.remove(p)
            except OSError as e:
                print(f"[FBCprRunner] WARN: could not remove {p}: {e}", flush=True)
                continue
            replay_p = _replay_sibling_path(p)
            if os.path.exists(replay_p):
                try:
                    os.remove(replay_p)
                except OSError as e:
                    print(f"[FBCprRunner] WARN: could not remove {replay_p}: {e}",
                          flush=True)

    def _mirror_checkpoint_to_s3(self, local_path: str) -> None:
        """Upload ``local_path`` to the configured S3 prefix in a BACKGROUND
        daemon thread, maintaining ONE rolling object (``s3_ckpt_name``). The
        file is COPIED to a temp path first so the in-flight upload is immune to
        the checkpoint being pruned/overwritten by later iters. Non-blocking;
        upload failures are logged, never raised. If a previous upload is still
        running, this one is skipped (the next save will catch up) so threads
        don't pile up on a slow link.
        """
        if not self.s3_ckpt_uri or not self._is_head:
            return
        if self._s3_thread is not None and self._s3_thread.is_alive():
            print("[FBCprRunner] S3 mirror: previous upload still running — "
                  f"skipping {os.path.basename(local_path)} (will catch up next save).",
                  flush=True)
            return
        if not os.path.exists(local_path):
            return
        # Snapshot the file so the upload is decoupled from disk churn.
        staged = local_path + ".s3upload.tmp"
        try:
            shutil.copy2(local_path, staged)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[FBCprRunner] WARN: S3 mirror stage copy failed: {e}", flush=True)
            return
        upload_name = self.s3_ckpt_name
        if self._is_resumed_run:
            stem, ext = os.path.splitext(upload_name)
            upload_name = f"{stem}_resume{ext}"
        dest = f"{self.s3_ckpt_uri}/{upload_name}"

        def _worker():
            t0 = time.time()
            try:
                # Do not inherit AWS_PROFILE or pass --profile: the cluster's
                # confidential credential provider is resolved through the
                # default AWS credential chain. Keep this change local to the
                # subprocesses instead of mutating the training process.
                aws_env = os.environ.copy()
                aws_env.pop("AWS_PROFILE", None)
                subprocess.run(
                    ["aws", "configure", "set", "region", "us-east-1"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    env=aws_env,
                )
                subprocess.run(
                    ["aws", "s3", "cp", staged, dest],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    env=aws_env,
                )
                print(f"[FBCprRunner] S3 mirror: uploaded -> {dest} "
                      f"({time.time() - t0:.1f}s)", flush=True)
            except subprocess.CalledProcessError as e:
                err = (e.stderr or b"").decode(errors="replace")[-400:]
                print(f"[FBCprRunner] WARN: S3 mirror upload failed: {err}", flush=True)
            except Exception as e:  # pragma: no cover - defensive
                print(f"[FBCprRunner] WARN: S3 mirror upload error: {e}", flush=True)
            finally:
                try:
                    os.remove(staged)
                except OSError:
                    pass

        self._s3_thread = threading.Thread(target=_worker, name="s3-ckpt-mirror", daemon=True)
        self._s3_thread.start()

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        *,
        load_replay: bool = False,
    ) -> dict:
        """Load a checkpoint. Set ``load_replay=True`` to also restore the
        replay buffer from the sibling ``.replay.pt`` file.

        IMPORTANT: if the replay buffer is NOT restored (either because
        ``load_replay=False`` or the sibling file is missing), the runner
        rewinds ``local_timesteps`` and ``_last_eval_step`` so the
        warmup and eval gates both re-fire on resume. Without this:

          - warmup is skipped (``local_timesteps >> num_seed_steps``) so
            the policy fills the empty replay with on-policy rollouts
            only — no uniform-random exploration to re-seed the buffer.
          - eval doesn't fire again until ``eval_every_steps`` LOCAL env-
            steps have accumulated post-resume, which is counter-intuitive
            for a just-loaded checkpoint.
          - the disc sees a tiny replay of repeated transitions and
            relearns its separator on a degenerate distribution, which
            manifests as disc_loss climbing visibly after resume.

        The raw and logging step counters are preserved regardless; only the
        cadence gates are rewound.
        """
        # Deserialize on CPU. Loading a resumed XL checkpoint directly onto CUDA
        # temporarily holds the checkpoint's model + Adam tensors alongside the
        # already constructed model/optimizers, which can OOM one rank while the
        # others later wait in a collective. load_state_dict copies/casts each
        # model and optimizer tensor onto its owning parameter device.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        saved_policy_cfg = ckpt.get("policy_cfg", {})
        saved_normalized_forward = bool(
            saved_policy_cfg.get("forward_gamma_normalized_output", False)
        )
        current_normalized_forward = bool(
            getattr(self.policy, "forward_gamma_normalized_output", False)
        )
        if saved_normalized_forward != current_normalized_forward:
            # Raw-F and G=(1-gamma)F checkpoints have identical tensor shapes
            # but incompatible semantics. Preserve the saved contract.
            self.policy.forward_gamma_normalized_output = (
                saved_normalized_forward
            )
            if hasattr(self.policy, "cfg"):
                self.policy.cfg.forward_gamma_normalized_output = (
                    saved_normalized_forward
                )
            self.policy_cfg["forward_gamma_normalized_output"] = (
                saved_normalized_forward
            )
            print(
                "[FBCprRunner] preserving checkpoint forward-map semantics: "
                f"normalized_output={saved_normalized_forward}.",
                flush=True,
            )
        model_sd = ckpt["model"]
        # Align checkpoint keys to the current policy's prefix scheme.
        # DDP adds ``.module.``; torch.compile adds ``._orig_mod.``. The
        # checkpoint may have either, both, or neither (depending on
        # how it was saved); the current model may also have either.
        # Translate to whatever the current policy expects.
        cur_keys = set(self.policy.state_dict().keys())
        cur_has_orig_mod = any("._orig_mod." in k for k in cur_keys)
        cur_has_module = any(".module." in k for k in cur_keys)

        def _align(k: str) -> str:
            # First strip both prefixes to get the canonical name.
            base = k.replace(".module.", ".").replace("._orig_mod.", ".")
            if base in cur_keys:
                return base
            # Then add prefixes the current model expects. Try insertion
            # at common patterns until we find a match.
            for prefix in ("._orig_mod.", ".module.", "._orig_mod.module.",
                           ".module._orig_mod."):
                # Insert after each top-level submodule name
                # (e.g. _backward_map. -> _backward_map._orig_mod.)
                parts = base.split(".")
                for i in range(1, len(parts)):
                    cand = ".".join(parts[:i]) + prefix + ".".join(parts[i:])
                    if cand in cur_keys:
                        return cand
            return k  # fallback: keep as-is, will be reported as unexpected

        if cur_has_orig_mod or cur_has_module or any(
            ".module." in k or "._orig_mod." in k for k in model_sd
        ):
            fixed_sd = {_align(k): v for k, v in model_sd.items()}
            n_fixed = sum(1 for k in fixed_sd if k not in model_sd)
            if n_fixed:
                print(f"[FBCprRunner] aligned {n_fixed} state_dict keys to "
                      f"current model's prefix scheme "
                      f"(orig_mod={cur_has_orig_mod}, module={cur_has_module}).",
                      flush=True)
            model_sd = fixed_sd
        # Non-strict load so legacy checkpoints (pre-reconstruction-head,
        # old MA with D(s,s'), etc.) still resume. Drop shape-mismatched
        # keys so the affected modules reinitialize cleanly.
        cur_sd = self.policy.state_dict()
        shape_mismatch = [
            k for k in list(model_sd.keys())
            if k in cur_sd and model_sd[k].shape != cur_sd[k].shape
        ]
        if shape_mismatch:
            print(f"[FBCprRunner] dropping {len(shape_mismatch)} shape-mismatched "
                  f"keys (will reinit): {shape_mismatch[:6]}"
                  f"{' ...' if len(shape_mismatch) > 6 else ''}", flush=True)
            for k in shape_mismatch:
                del model_sd[k]
        missing, unexpected = self.policy.load_state_dict(model_sd, strict=False)
        # Treat shape-mismatched keys as missing (so optimizer logic below knows).
        missing = list(missing) + shape_mismatch
        new_head_keys = [k for k in missing if "_reconstruction_head" in k]
        other_missing = [k for k in missing if "_reconstruction_head" not in k]
        if new_head_keys:
            print(
                f"[FBCprRunner] checkpoint predates reconstruction head — "
                f"{len(new_head_keys)} head params randomly initialised.",
                flush=True,
            )
        if other_missing:
            print(f"[FBCprRunner] WARN missing state_dict keys: {other_missing[:8]}"
                  f"{' ...' if len(other_missing) > 8 else ''}", flush=True)
        if unexpected:
            print(f"[FBCprRunner] WARN unexpected state_dict keys: {unexpected[:8]}"
                  f"{' ...' if len(unexpected) > 8 else ''}", flush=True)
        # Optimizer state is loaded by param-group POSITION; torch only checks
        # the param COUNT, not shapes. If a module was reinitialised (any of its
        # keys landed in missing / unexpected / shape_mismatch — e.g. an arch
        # swap like MLP->transformer actor, or the 256->384 z / 4->9 history
        # resize, or the MLP->linear recon head), loading its old optimizer state
        # by position can silently mismatch moment buffers (or crash on the first
        # fused step). So skip the optimizer for EVERY reinitialised module, not
        # just the manifold attractor. Map each optimizer -> the module-key
        # substrings it owns (backward_optimizer owns B AND the recon head).
        reinit_keys = set(missing) | set(unexpected) | set(shape_mismatch)
        _opt_modules = {
            "actor_optimizer": ("_actor",),
            "critic_optimizer": ("_critic",),
            "aux_critic_optimizer": ("_aux_critic",),
            "forward_optimizer": ("_forward_map",),
            "backward_optimizer": ("_backward_map", "_reconstruction_head"),
            "discriminator_optimizer": ("_discriminator",),
            "entropy_critic_optimizer": ("_entropy_critic",),
            "manifold_attractor_optimizer": ("_manifold_attractor", "manifold_attractor"),
        }

        def _opt_reinitialised(opt_name: str) -> bool:
            subs = _opt_modules.get(opt_name, ())
            return any(any(s in k for s in subs) for k in reinit_keys)

        if load_optimizer and "optimizers" in ckpt:
            for name, sd in ckpt["optimizers"].items():
                if _opt_reinitialised(name):
                    print(f"[FBCprRunner] skipping {name} (its module was "
                          f"reinitialised — arch/shape change) — fresh optimizer.",
                          flush=True)
                    continue
                opt = getattr(self.alg, name, None)
                if opt is None:
                    continue
                try:
                    opt.load_state_dict(sd)
                except (ValueError, RuntimeError) as e:
                    print(
                        f"[FBCprRunner] WARN optimizer '{name}' state did not "
                        f"match current params (probably new head attached) — "
                        f"leaving at fresh init. ({e})",
                        flush=True,
                    )
            # ``Adam.load_state_dict`` overwrites each param_group's ``lr``
            # with the saved value — so the LR scaling (``sqrt(world_size) *
            # sqrt(batch_size / 1024)``, 0.25× disc damping, etc.) computed
            # fresh at runner init gets clobbered on resume. Re-apply the
            # current-cfg LRs here so cfg changes (new world_size, new
            # batch_size, new disc-damping factor, LR-anneal schedule)
            # take effect immediately on resume. The Adam momentum buffers
            # loaded from ckpt remain intact.
            if hasattr(self.alg, "_start_lrs"):
                opts_by_name = (
                    ("actor", getattr(self.alg, "actor_optimizer", None)),
                    ("critic", getattr(self.alg, "critic_optimizer", None)),
                    ("aux_critic", getattr(self.alg, "aux_critic_optimizer", None)),
                    ("f", getattr(self.alg, "forward_optimizer", None)),
                    ("b", getattr(self.alg, "backward_optimizer", None)),
                    ("discriminator", getattr(self.alg, "discriminator_optimizer", None)),
                    ("entropy_critic", getattr(self.alg, "entropy_critic_optimizer", None)),
                    ("manifold_attractor", getattr(self.alg, "manifold_attractor_optimizer", None)),
                )
                reapplied: dict[str, float] = {}
                for name, opt in opts_by_name:
                    if opt is None or name not in self.alg._start_lrs:
                        continue
                    lr = float(self.alg._start_lrs[name])
                    for g in opt.param_groups:
                        g["lr"] = lr
                    reapplied[name] = lr
                if reapplied:
                    pretty = "  ".join(f"{k}={v:.3g}" for k, v in reapplied.items())
                    print(
                        f"[FBCprRunner] reapplied start LRs after resume "
                        f"(Adam momentums preserved): {pretty}",
                        flush=True,
                    )
            fb_reinitialised = any(
                "_forward_map" in key or "_backward_map" in key
                for key in reinit_keys
            )
            if not fb_reinitialised:
                self.alg.load_training_state_dict(
                    ckpt.get("algorithm_state", {})
                )
        self.current_learning_iteration = ckpt.get("iter", 0)
        self.tot_timesteps = ckpt.get("tot_timesteps", 0)
        self._local_timesteps = ckpt.get("local_timesteps", 0)
        if "log_timesteps" in ckpt:
            self.log_timesteps = int(ckpt["log_timesteps"])
        else:
            # Legacy checkpoints did not persist a logging-only counter or
            # world size. Infer the old world size from global/local steps, or
            # from completed iterations when periodic saves still have a stale
            # local counter. Fall back to the current world size.
            saved_world_size = ckpt.get("world_size")
            completed_iterations = int(ckpt.get("iter", -1)) + 1
            iter_local_timesteps = (
                completed_iterations * self.env.num_envs * self.num_steps_per_env
            )
            for candidate_local_steps in (
                self._local_timesteps,
                iter_local_timesteps,
            ):
                if saved_world_size is not None or candidate_local_steps <= 0:
                    continue
                ratio = self.tot_timesteps / candidate_local_steps
                rounded_ratio = round(ratio)
                if 1 <= rounded_ratio <= 4096 and abs(ratio - rounded_ratio) < 1e-6:
                    saved_world_size = rounded_ratio
            saved_world_size = max(int(saved_world_size or self.gpu_world_size), 1)
            log_ws_cap = int(self.cfg.get("log_env_steps_world_size_cap", 0))
            saved_log_world_size = (
                min(saved_world_size, log_ws_cap)
                if log_ws_cap > 0
                else saved_world_size
            )
            self.log_timesteps = round(
                self.tot_timesteps * saved_log_world_size / saved_world_size
            )
        self._last_eval_step = ckpt.get("last_eval_step", self._last_eval_step)
        priority_state = ckpt.get("expert_priority_state")
        if (
            priority_state is not None
            and hasattr(self.expert_buffer, "load_priority_state_dict")
        ):
            restored = self.expert_buffer.load_priority_state_dict(
                priority_state
            )
            if not restored:
                print(
                    "[FBCprRunner] expert priority state does not match this "
                    "dataset shard; using fresh priorities.",
                    flush=True,
                )

        replay_restored = False
        if load_replay:
            replay_path = _replay_sibling_path(path)
            if os.path.exists(replay_path):
                rdict = torch.load(replay_path, map_location="cpu", weights_only=False)
                self.replay_buffer.load_state_dict(rdict["replay"])
                replay_restored = True
                print(f"[FBCprRunner] loaded replay from {replay_path}", flush=True)
            else:
                print(f"[FBCprRunner] WARN: load_replay=True but no {replay_path}",
                      flush=True)

        if not replay_restored:
            # Replay is empty. Rewind the cadence gates so warmup re-runs and
            # eval fires again early. Keep self.tot_timesteps so the wandb
            # x-axis continues smoothly.
            self._local_timesteps = 0
            self._last_eval_step = 0
            # Refill the empty buffer BEFORE updates begin, using the ACTUAL
            # TRAINING ROLLOUT (the resumed policy + its exploration/z-context),
            # NOT uniform-random actions — the policy is already trained, so
            # random garbage would pollute the buffer. We (a) delay the update
            # gate (``_delay_updates_until``) to resume_num_seed_steps and (b)
            # set the random-action window (``_random_seed_until``) to 0. So the
            # resumed policy collects fully on-policy until resume_num_seed_steps
            # transitions accumulate, then updates start on a well-filled buffer.
            # Falls back to num_seed_steps when resume_num_seed_steps is unset.
            _resume_seed = self.alg_cfg.get("resume_num_seed_steps", None)
            if _resume_seed is not None and int(_resume_seed) > self.num_seed_steps:
                self._delay_updates_until = int(_resume_seed)
                # Trained policy -> collect fully on-policy, NO random warmup.
                self._random_seed_until = 0
                print(
                    f"[FBCprRunner] no-replay resume: on-policy collect (0 random) "
                    f"until {self._delay_updates_until} per-rank env-steps before "
                    f"updates begin (resume_num_seed_steps).",
                    flush=True,
                )
            print(
                f"[FBCprRunner] Replay not restored — rewinding "
                f"local_timesteps=0 and _last_eval_step=0. Raw/logging step "
                f"counters kept at {self.tot_timesteps}/{self.log_timesteps}.",
                flush=True,
            )
        # Reset torch.compile cache after loading — prevents compiled graph
        # guard mismatches from causing permanent eager-mode fallback.
        try:
            torch._dynamo.reset()
        except Exception:
            pass

        self._checkpoint_loaded = True
        self._checkpoint_requires_parameter_sync = bool(reinit_keys)
        self._is_resumed_run = True

        return ckpt.get("infos", {})

    # --- train/eval toggles ---------------------------------------------- #

    def train_mode(self) -> None:
        self.policy.train(True)

    def eval_mode(self) -> None:
        self.policy.train(False)

    def get_inference_policy(self, device: str | None = None):
        if device is not None:
            self.policy.to(device)
        self.policy.train(False)
        return self.policy

    def add_git_repo_to_log(self, file: str) -> None:
        """No-op stub — rsl_rl's BaseRunner signature. We don't track git state."""
        return None


##########################
# FBCprCondRunner — adds the ``measure_cond`` obs group
##########################


class FBCprCondRunner(FBCprRunner):
    """FBCpr runner variant for :class:`FBCprCond` / :class:`FBCprCondPolicy`.

    Identical to :class:`FBCprRunner` except:
      * instantiates :class:`FBCprCondPolicy` + :class:`FBCprCond`,
      * adds a default ``measure_cond`` entry to the obs-key-groups map
        so the env's ``measure_cond`` term is collected into its own
        group in the Dict obs space (users can still override this via
        ``alg_cfg['obs_key_groups']``).
    """

    _POLICY_CLS = FBCprCondPolicy
    _ALGO_CLS = FBCprCond
    _NET_CFG_CLS = FBCprCondNetworkCfg
    _ALGO_CFG_CLS = FBCprCondAlgorithmCfg

    # Extend the default obs-key-groups with a ``measure_cond`` group. The
    # env is expected to define an obs term of the same name on the
    # critic (and optionally policy) side. If the env uses a different
    # term name, override via ``alg_cfg['obs_key_groups']``.
    _BFM_KEY_GROUPS_DEFAULT: dict[str, tuple[str, ...]] = {
        **FBCprRunner._BFM_KEY_GROUPS_DEFAULT,
        "measure_cond": ("measure_cond",),
    }


class AnchoredFBCprRunner(FBCprRunner):
    """FB-CPR runner for BFM-One-Anchored (Global-through-Anchoring).

    Single-B formulation: uses the STANDARD policy (one backward map B that
    additionally reads ``anchored_pose`` via its backward_input_keys), plus the
    anchored algorithm (per-row anchor relabel of the obs) and world-pose replay
    storage / per-episode anchor setting. Adds the ``anchored_pose`` obs-key
    group (the env emits A^-1 g_t).
    """

    _POLICY_CLS = FBCprAuxPolicy
    _ALGO_CLS = AnchoredFBCprAux
    _NET_CFG_CLS = FBCprNetworkCfg
    _ALGO_CFG_CLS = FBCprAuxAlgorithmCfg

    _BFM_KEY_GROUPS_DEFAULT: dict[str, tuple[str, ...]] = {
        **FBCprRunner._BFM_KEY_GROUPS_DEFAULT,
        "anchored_pose": ("anchored_pose",),
    }
