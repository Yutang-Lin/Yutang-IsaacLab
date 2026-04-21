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

import os
import statistics
import time
from collections import deque
from typing import Any, Dict

import torch

from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

from isaaclab_rl.rsl_rl.algorithms.fb_cpr import FBCprAux, FBCprAuxAlgorithmCfg
from isaaclab_rl.rsl_rl.modules.fb_cpr_policy import (
    FBCprAuxPolicy,
    FBCprNetworkCfg,
)
from isaaclab_rl.rsl_rl.storage.fb_cpr_storage import (
    FBCprExpertBuffer,
    FBCprReplayBuffer,
)

__all__ = ["FBCprRunner"]


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
            # Bind PyTorch's current device BEFORE init_process_group. Torch 2.3+
            # NCCL init is eager and will create a communicator per visible GPU
            # if it can't resolve a unique device, leaking ~400 MiB ctx onto
            # every peer GPU on the node.
            torch.cuda.set_device(self.gpu_local_rank)
            torch.distributed.init_process_group(
                backend="nccl",
                rank=self.gpu_global_rank,
                world_size=self.gpu_world_size,
                device_id=torch.device(f"cuda:{self.gpu_local_rank}"),
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
        self.policy = FBCprAuxPolicy(self.obs_space, action_dim=self.action_dim, cfg=net_cfg)

        # --- Algorithm --------------------------------------------------
        algo_cfg = self._build_algo_cfg(self.alg_cfg)
        self.alg = FBCprAux(self.policy, cfg=algo_cfg, device=self.device)

        # --- Expert buffer ---------------------------------------------
        expert_path = self.alg_cfg.get("expert_dataset_path")
        expert_device = self.alg_cfg.get("expert_dataset_device", "cuda")
        # Resolve bare "cuda" to the rank-local GPU so every rank doesn't load
        # the full expert dataset onto cuda:0. torch.load(map_location="cuda")
        # respects current device, but being explicit prevents any accidental
        # cuda:0 allocation during subsequent ``.to()`` / sampling.
        if expert_device == "cuda":
            expert_device = f"cuda:{self.gpu_local_rank}"
        self.expert_buffer = FBCprExpertBuffer(
            pt_path=expert_path,
            seq_length=net_cfg.seq_length,
            device=expert_device,
        )
        # Forward to the env so RSI can pull from it.
        if hasattr(self.env_unwrapped, "set_expert_buffer"):
            self.env_unwrapped.set_expert_buffer(self.expert_buffer)

        # --- Replay buffer ---------------------------------------------
        aux_reward_names = list(algo_cfg.aux_rewards_scaling.keys())
        self.replay_buffer = FBCprReplayBuffer(
            capacity=int(self.alg_cfg.get("replay_capacity", 5_120_000)),
            obs_space=self.obs_space,
            action_dim=self.action_dim,
            z_dim=self.policy.z_dim,
            aux_reward_names=aux_reward_names,
            device=self.alg_cfg.get("replay_device", "cpu"),
        )

        # --- Seed / rhythm controls ------------------------------------
        self.num_seed_steps = int(self.alg_cfg.get("num_seed_steps", 10_240))
        self.num_agent_updates = int(self.alg_cfg.get("num_agent_updates", 16))
        self.update_agent_every = int(self.alg_cfg.get("update_agent_every", 1024))
        self.save_interval = int(self.cfg.get("save_interval", 50))

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
        self.eval_update_priorities = bool(self.alg_cfg.get("eval_update_priorities", True))
        self.eval_priority_min = float(self.alg_cfg.get("eval_priority_min", 0.5))
        self.eval_priority_max = float(self.alg_cfg.get("eval_priority_max", 2.0))
        self.eval_priority_scale = float(self.alg_cfg.get("eval_priority_scale", 2.0))
        self.eval_priority_mode = str(self.alg_cfg.get("eval_priority_mode", "exp"))
        self._last_eval_step = 0

        # --- Logging ---------------------------------------------------
        self.num_steps_per_env = int(self.cfg.get("num_steps_per_env", 1))
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0

    # --- BFM-Zero obs-space composition ------------------------------- #

    # Maps the 4 BFM-agent obs keys to the per-term names we expect to
    # find in the env's observation_cfg. The env must define obs terms
    # with these names; the runner concatenates them in the same order
    # inside each group to match the expert-dataset layout.
    _BFM_KEY_GROUPS: dict[str, tuple[str, ...]] = {
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

    @staticmethod
    def _build_network_cfg(policy_cfg: dict) -> FBCprNetworkCfg:
        cfg = FBCprNetworkCfg()
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

    @staticmethod
    def _build_algo_cfg(alg_cfg: dict) -> FBCprAuxAlgorithmCfg:
        cfg = FBCprAuxAlgorithmCfg(aux_rewards_scaling=dict(alg_cfg.get("aux_rewards_scaling", {})))
        for k, v in alg_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

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

        # DDP: broadcast the rank-0 parameters + running stats to every rank
        # so every worker starts from the same weights. Must run BEFORE the
        # first forward pass; the algorithm's ``broadcast_parameters`` is a
        # no-op when ``is_distributed`` is False.
        if self.is_distributed:
            print(f"[FBCprRunner] Synchronizing parameters for rank {self.gpu_global_rank}...", flush=True)
            self.alg.broadcast_parameters()

        obs_flat, extras = self.env.get_observations()
        obs_dict = self._obs_to_device(obs_flat, extras)
        step_count = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)

        # Per-env z context.
        z_context = self.alg.maybe_update_rollout_context(
            z=None, step_count=step_count, expert_buffer=self.expert_buffer
        )

        rewbuffer: deque[float] = deque(maxlen=500)
        lenbuffer: deque[float] = deque(maxlen=500)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_ep_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # BFM-style done_prev mask: envs that reset on the PREVIOUS step have
        # a fresh post-reset obs in ``obs_dict`` that should NOT be paired
        # with the current-iter ``new_obs`` as a valid TD transition (the
        # pair would span a reset boundary and poison the TD target).
        # BFM filters via ``indexes = ~done`` before ``replay_buffer.extend``
        # (train.py:458, 466, 478). We mirror this by writing only the
        # transitions from envs NOT flagged in ``done_prev``.
        done_prev = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        best_reward = -float("inf")

        steps_since_last_update = 0
        total_metrics: Dict[str, torch.Tensor] | None = None
        num_metrics_updates = 0

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
                env_steps = int(self.tot_timesteps)
                for k, v in eval0.items():
                    self.writer.add_scalar(f"{k}_initial", v, env_steps)
                print(
                    f"[FBCprRunner] initial eval: "
                    f"mpjpe_mm={eval0.get('Eval/mpjpe_mm', 0):.1f} "
                    f"emd={eval0.get('Eval/emd', 0):.3g} "
                    f"tracking_success={eval0.get('Eval/tracking_success', 0):.3g} "
                    f"num_motions={int(eval0.get('Eval/num_motions', 0))}",
                    flush=True,
                )
            # After the initial eval we refresh obs_dict/z_context because
            # the eval rolled + restored the env; the cached obs_dict we
            # already took (line 369) is still from before the eval so it
            # is still valid, but z_context may have been touched by the
            # internal eval act() calls. Recompute to be safe.
            z_context = self.alg.maybe_update_rollout_context(
                z=None, step_count=step_count, expert_buffer=self.expert_buffer,
            )

        # Track LOCAL env-steps (per-rank) for warmup / update cadence / eval
        # cadence. Each rank has its own replay buffer, so
        # ``num_seed_steps``/``update_agent_every``/``eval_every_steps`` are
        # per-rank budgets. BFM has this too: every rank runs its own train
        # loop and only the wall-clock metrics are world-scaled.
        # ``self.tot_timesteps`` remains the GLOBAL (env_steps × world_size)
        # counter used only for logging / reporting.
        local_timesteps = getattr(self, "_local_timesteps", 0)

        for it in range(start_iter, tot_iter):
            start = time.time()
            # ----- rollout -----
            with torch.inference_mode():
                self.policy.eval()
                for _ in range(self.num_steps_per_env):
                    warmup = local_timesteps < self.num_seed_steps
                    if warmup:
                        actions = torch.zeros(
                            self.env.num_envs, self.action_dim, device=self.device
                        ).uniform_(-1.0, 1.0)
                    else:
                        actions = self.policy.act(obs_dict, z_context, mean=False)

                    new_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    new_obs = self._obs_to_device(new_obs, infos)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

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

                    # Drop transitions for envs that reset last step (``done_prev``):
                    # ``obs_dict`` for those envs is the post-reset observation,
                    # so pairing it with the current step's ``new_obs`` as an
                    # (s, a, s') tuple would span a reset boundary. BFM
                    # filters the same way via ``indexes = ~done``
                    # (train.py:458, 466, 478).
                    keep = ~done_prev
                    if bool(keep.any().item()):
                        keep_idx = keep.nonzero(as_tuple=False).view(-1)
                        batch_obs = {k: v[keep_idx] for k, v in obs_dict.items()}
                        batch_next_obs = {k: v[keep_idx] for k, v in new_obs.items()}
                        batch = {
                            "observation": batch_obs,
                            "action": actions[keep_idx],
                            "z": z_context[keep_idx],
                            "next": {
                                "observation": batch_next_obs,
                                "terminated": terminated[keep_idx],
                            },
                            "aux_rewards": {k: v[keep_idx] for k, v in aux_rewards_dict.items()},
                        }
                        # Safety net: any iter where stored obs / aux_rewards go
                        # NaN or inf, print once and continue. Matches BFM's
                        # implicit fail-loud behaviour without killing the run.
                        if (self.tot_timesteps // self.env.num_envs) % 500 == 0:
                            for k, v in batch_obs.items():
                                if torch.isnan(v).any() or torch.isinf(v).any():
                                    print(f"[WARN env_steps={self.tot_timesteps}] "
                                          f"obs[{k}] has nan={int(torch.isnan(v).sum())} "
                                          f"inf={int(torch.isinf(v).sum())}", flush=True)
                            for k, v in batch["aux_rewards"].items():
                                if torch.isnan(v).any() or torch.isinf(v).any():
                                    print(f"[WARN env_steps={self.tot_timesteps}] "
                                          f"aux_rewards[{k}] has nan/inf", flush=True)
                        self.replay_buffer.add(batch)

                    # Book-keeping
                    cur_reward_sum += rewards
                    cur_ep_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_ep_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_ep_length[new_ids] = 0

                    # Update done_prev (for next iter's write-filter).
                    done_prev = dones.bool()

                    # Update per-env step_count (reset on done).
                    step_count = step_count + 1
                    step_count = torch.where(dones.bool(), torch.zeros_like(step_count), step_count)

                    # Update z context for next step.
                    z_context = self.alg.maybe_update_rollout_context(
                        z=z_context, step_count=step_count, expert_buffer=self.expert_buffer
                    )

                    obs_dict = new_obs

                    # Local = per-rank steps (drives warmup + update cadence).
                    # Global = world-scaled (for logging only).
                    local_timesteps += self.env.num_envs
                    self.tot_timesteps += self.env.num_envs * self.gpu_world_size
                    steps_since_last_update += self.env.num_envs

            collection_time = time.time() - start
            start = time.time()

            # ----- updates -----
            loss_dict: Dict[str, float] = {}
            warmup_flag = local_timesteps < self.num_seed_steps
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
                expert_chunks = self.expert_buffer.sample_chunks(
                    batch_size, self.num_agent_updates, target_device=self.device,
                )
                replay_dict = {
                    "train": _PrefetchedSampler(train_chunks),
                    "expert_slicer": _PrefetchedSampler(expert_chunks),
                }
                for _ in range(self.num_agent_updates):
                    metrics = self.alg.update(replay_dict, step=int(self.tot_timesteps))
                    if total_metrics is None:
                        total_metrics = {k: v.float().detach().clone() for k, v in metrics.items()}
                        num_metrics_updates = 1
                    else:
                        for k, v in metrics.items():
                            total_metrics[k] = total_metrics.get(k, torch.zeros_like(v.float())) + v.float().detach()
                        num_metrics_updates += 1

                # Single running-stat sync per iter (was per-update = 16×).
                if self.is_distributed:
                    self.alg._sync_running_stats()

                for k, v in total_metrics.items():
                    loss_dict[k] = float(v.mean().item()) / max(num_metrics_updates, 1)
                total_metrics = None
                num_metrics_updates = 0

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
                            self.save(os.path.join(self.log_dir, "model_best.pt"))
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            if self._is_head and it == start_iter and self.log_dir is not None:
                import rsl_rl
                store_code_state(self.log_dir, [rsl_rl.__file__])

        self._local_timesteps = local_timesteps
        if self._is_head and self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

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
        if not getattr(self.expert_buffer, "supports_reset_states", False):
            return None
        num_envs = self.env.num_envs
        num_motions = self.expert_buffer.num_unique_motions
        if num_motions == 0:
            return None

        env_u = self.env_unwrapped
        if not (hasattr(env_u, "snapshot_state") and hasattr(env_u, "restore_state")):
            return None

        snap = env_u.snapshot_state()
        try:
            # --- assign a motion id to each env (cycle) ---
            motion_of_env = torch.arange(num_envs, device=self.device) % num_motions

            # --- per-motion windows of length L = eval_rollout_length+1 ---
            L = int(self.eval_rollout_length) + 1
            seq_length = int(self.policy.seq_length)
            action_dim = int(self.action_dim)
            num_joints = int(self.expert_buffer.num_joints)

            # Pre-encode z per motion via B + rolling seq_length mean.
            z_per_motion: list[torch.Tensor] = []
            for m in range(num_motions):
                win = self.expert_buffer.get_motion_window(m, num_frames=L)
                next_obs_dict = {
                    "state": win["state"][1:].to(self.device, non_blocking=True),
                    "privileged_state": win["privileged_state"][1:].to(self.device, non_blocking=True),
                    "last_action": win["last_action"][1:].to(self.device, non_blocking=True),
                    "history_actor": win["history_actor"][1:].to(self.device, non_blocking=True),
                }
                z = self.policy.backward_map(next_obs_dict)   # [L-1, z_dim]
                # rolling mean over seq_length-window (BFM tracking_inference)
                for s in range(z.shape[0]):
                    end = min(s + seq_length, z.shape[0])
                    z[s] = z[s:end].mean(dim=0)
                z = self.policy.project_z(z)
                z_per_motion.append(z)

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
            # Write the motion-aligned initial state directly to the sim.
            joint_pos_full = torch.zeros(num_envs, env_u.robot.data.joint_pos.shape[1], device=self.device)
            joint_vel_full = torch.zeros_like(joint_pos_full)
            joint_order_t = torch.as_tensor(env_u.joint_order, device=self.device, dtype=torch.long)
            joint_pos_full[:, joint_order_t] = jp0
            joint_vel_full[:, joint_order_t] = jv0
            env_ids_all = env_u._ALL_INDICES
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

            # --- rollout ---
            obs_flat, extras = self.env.get_observations()
            obs_dict = self._obs_to_device(obs_flat, extras)

            # Per-env joint_pos log (for MPJPE) and dof_pos_dev log (for EMD).
            jp_log = torch.zeros(num_envs, L, num_joints, device=self.device)
            # dof_pos_dev = first num_joints dims of the env's `state` obs key
            dpd_log = torch.zeros(num_envs, L, num_joints, device=self.device)
            jp_log[:, 0] = env_u.joint_pos[:, joint_order_t]
            dpd_log[:, 0] = obs_dict["state"][:, :num_joints]
            for t in range(1, L):
                # Pack z for each env at time t (cap at motion length).
                z_batch = torch.zeros(num_envs, self.policy.z_dim, device=self.device)
                for m in range(num_motions):
                    zm = z_per_motion[m]
                    if zm.shape[0] == 0:
                        continue
                    idx = min(t - 1, zm.shape[0] - 1)
                    mask = motion_of_env == m
                    if mask.any():
                        z_batch[mask] = zm[idx]
                action = self.policy.act(obs_dict, z_batch, mean=True)
                new_obs, _, _, infos = self.env.step(action.to(self.env.device))
                obs_dict = self._obs_to_device(new_obs, infos)
                jp_log[:, t] = env_u.joint_pos[:, joint_order_t]
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
                # pairwise L2 distance matrix, then uniform-mass OT.
                cost = torch.cdist(agent_seq, ref_seq, p=2).detach().cpu().numpy()
                a = _np.ones(cost.shape[0]) / cost.shape[0]
                b = _np.ones(cost.shape[1]) / cost.shape[1]
                try:
                    emd_val = float(_ot.emd2(a, b, cost, numItermax=100_000))
                except Exception:
                    emd_val = float("nan")
                emd_per_motion[m] = emd_val
                count_per_motion[m] = 1.0

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

            return {
                "Eval/mpjpe_mm": mean_mpjpe,
                "Eval/emd": mean_emd,
                "Eval/tracking_success": success,
                "Eval/num_motions": float(int(valid.sum().item())),
            }
        finally:
            if hasattr(env_u, "_eval_mode"):
                env_u._eval_mode = False
            env_u.restore_state(snap)

    # --- utilities ------------------------------------------------------- #

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
        env_steps = int(self.tot_timesteps)
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
            "Eval/mpjpe_mm",
            "Eval/tracking_success",
        )
        for k in loss_keys_to_print:
            if k in loss_dict:
                summary += f"  {k}={loss_dict[k]:.3g}"
        print(summary, flush=True)

    # --- checkpoint I/O -------------------------------------------------- #

    def save(self, path: str, infos: Any | None = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizers": self.alg.optimizer_dict,
                "iter": self.current_learning_iteration,
                "tot_timesteps": self.tot_timesteps,
                "local_timesteps": getattr(self, "_local_timesteps", 0),
                "last_eval_step": self._last_eval_step,
                "infos": infos or {},
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["model"])
        if load_optimizer and "optimizers" in ckpt:
            for name, sd in ckpt["optimizers"].items():
                opt = getattr(self.alg, name, None)
                if opt is not None:
                    opt.load_state_dict(sd)
        self.current_learning_iteration = ckpt.get("iter", 0)
        self.tot_timesteps = ckpt.get("tot_timesteps", 0)
        self._local_timesteps = ckpt.get("local_timesteps", 0)
        self._last_eval_step = ckpt.get("last_eval_step", self._last_eval_step)
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
