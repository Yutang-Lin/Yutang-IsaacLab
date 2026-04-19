# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner for the sparse-constraint successor tracking algorithm.

Extends BaseRunner to inject next-observation passing and constraint resampling
into the rollout loop, which SparseSuccessor requires for off-policy storage.
"""

from __future__ import annotations

import os
import statistics
import time
import torch

from collections import deque

from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

from isaaclab_rl.rsl_rl.runners.base_runner import BaseRunner
from isaaclab_rl.rsl_rl.algorithms.sparse_successor import SparseSuccessor


class SuccessorRunner(BaseRunner):
    """On-policy-style runner that collects rollouts into SuccessorStorage.

    Key differences from BaseRunner:
    - After env.step(), passes (next_obs, next_priv_obs) to alg.set_next_obs()
    - Skips GAE return computation (off-policy)
    - Handles constraint sampling lifecycle
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", **kwargs):
        # Force training_type to "rl" before parent init
        super().__init__(env, train_cfg, log_dir=log_dir, device=device, **kwargs)

        # Hand the algorithm's ExpertMotionBuffer to the env so reference-state
        # initialization (RSI) can pull from the same expert dataset used for
        # discriminator training. Must happen after the algorithm has loaded
        # its buffer (in SparseSuccessor.__init__) and before the first reset
        # triggered during rollout. BaseRunner runs env.reset() on demand in
        # learn(), so setting it here is always early enough.
        env_u = self.env.unwrapped
        alg_buffer = getattr(self.alg, "expert_buffer", None)
        if alg_buffer is not None and hasattr(env_u, "set_expert_buffer"):
            env_u.set_expert_buffer(alg_buffer)

    # ------------------------------------------------------------------
    # Logging — duplicate every metric against a BFM-style env-step x-axis
    # ------------------------------------------------------------------

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """BaseRunner.log logs everything against ``locs['it']``. Our ``num_steps_per_env``
        (5) is much smaller than PPO-style configs (~24+), which makes the iteration
        axis hard to compare to BFM-Zero's curves. We call the parent to keep all
        existing iteration-indexed metrics, then re-emit the same series against the
        cumulative env-step count (``Perf/env_steps_total``) so a user can switch the
        x-axis to ``env_step`` in wandb/tensorboard.
        """
        super().log(locs, width=width, pad=pad)
        if self.writer is None or self.disable_logs:
            return

        it = locs["it"]
        env_steps = int(self.tot_timesteps)  # updated inside super().log

        # Emit the absolute env-step count every iter — this is the new x-axis.
        self.writer.add_scalar("Perf/env_steps_total", env_steps, it)

        # Duplicate all losses against env_steps (second series under "LossVsEnvStep/").
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"LossVsEnvStep/{key}", value, env_steps)

        # Training rewards keyed by env_steps for direct comparison with BFM curves.
        import statistics as _stats
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "TrainVsEnvStep/mean_reward", _stats.mean(locs["rewbuffer"]), env_steps
            )
            self.writer.add_scalar(
                "TrainVsEnvStep/mean_episode_length", _stats.mean(locs["lenbuffer"]), env_steps
            )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # Initialize writer (reuse parent's writer init logic)
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)

        # Normalize
        obs = self.obs_normalizer(obs)
        if self.privileged_obs_type is not None:
            privileged_obs = self.privileged_obs_normalizer(privileged_obs)

        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=500)
        lenbuffer = deque(maxlen=500)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Initialize constraints from first privileged obs
        alg: SparseSuccessor = self.alg  # type: ignore
        alg._env_constraints = alg.sample_constraint_set_vectorized(privileged_obs, self.env.num_envs)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            if self.empirical_normalization:
                from isaaclab_rl.rsl_rl.runners.base_runner import _sync_normalizer
                _sync_normalizer(self.obs_normalizer, self.device)
                _sync_normalizer(self.privileged_obs_normalizer, self.device)

        infos = dict(meta_tensors={})
        best_reward = -float("inf")

        if not hasattr(self.alg, "unwrapped_env"):
            setattr(self.alg, "unwrapped_env", self.env.unwrapped)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()

            if hasattr(self.env.unwrapped, "pre_rollout"):
                self.env.unwrapped.pre_rollout()

            # ---- Rollout ----
            with torch.inference_mode():
                self.alg.policy.eval()

                for _ in range(self.num_steps_per_env):
                    actions = alg.act(obs, privileged_obs, infos=infos).clamp(*self.action_clip_range)

                    # Step environment
                    next_obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))  # type: ignore
                    next_obs, rewards, dones = (
                        next_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # Normalize next observations
                    next_obs = self.obs_normalizer(next_obs)
                    if self.privileged_obs_type is not None:
                        next_priv_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        next_priv_obs = next_obs

                    # Compute the BFM-style style-feature tensor for this step
                    # from the env (if it exposes the helper). This gets fed
                    # into the algorithm's vectorized snippet ring buffer.
                    style_features = None
                    env_u = self.env.unwrapped
                    if hasattr(env_u, "compute_style_features"):
                        style_features = env_u.compute_style_features().to(self.device)

                    # Process env step (records reward/done, pushes snippet frame)
                    alg.process_env_step(rewards, dones, infos, style_features=style_features)

                    # Pass next observations and commit to storage
                    alg.set_next_obs(next_obs, next_priv_obs)

                    # Handle resets
                    resets = infos.get("resets", dones == 1).to(self.device)
                    if hasattr(self.alg.policy, "reset"):
                        self.alg.policy.reset(resets)

                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                    # Update obs for next step
                    obs = next_obs
                    privileged_obs = next_priv_obs

            stop = time.time()
            collection_time = stop - start
            start = stop

            if hasattr(self.env.unwrapped, "post_rollout"):
                self.env.unwrapped.post_rollout()

            # Sync normalizers
            if self.is_distributed and self.empirical_normalization:
                from isaaclab_rl.rsl_rl.runners.base_runner import _sync_normalizer
                _sync_normalizer(self.obs_normalizer, self.device)
                _sync_normalizer(self.privileged_obs_normalizer, self.device)

            # ---- Train ----
            self.alg.policy.train()
            loss_dict = self.alg.update()

            # Reset-source breakdown (group 7 diag) — ask the env for RSI share.
            env_u = self.env.unwrapped
            if hasattr(env_u, "consume_reset_stats"):
                reset_stats = env_u.consume_reset_stats()
                loss_dict.update(reset_stats)

            if hasattr(self.env.unwrapped, "post_update"):
                env_loss_dict = self.env.unwrapped.post_update()
                if isinstance(env_loss_dict, dict):
                    loss_dict.update(env_loss_dict)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # ---- Reduce metrics across distributed ranks ----
            if self.is_distributed:
                import torch.distributed as dist
                has_data = len(rewbuffer) > 0
                local_reward = statistics.mean(rewbuffer) if has_data else 0.0
                local_ep_len = statistics.mean(lenbuffer) if has_data else 0.0
                local_count = float(len(rewbuffer))
                reward_t = torch.tensor([local_reward * local_count], device=self.device)
                ep_len_t = torch.tensor([local_ep_len * local_count], device=self.device)
                count_t = torch.tensor([local_count], device=self.device)
                dist.all_reduce(reward_t)
                dist.all_reduce(ep_len_t)
                dist.all_reduce(count_t)
                if count_t.item() > 0 and not self.disable_logs:
                    self.writer.add_scalar("Train/global_mean_reward", (reward_t / count_t).item(), it)
                    self.writer.add_scalar("Train/global_mean_episode_length", (ep_len_t / count_t).item(), it)

            # ---- Log ----
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    if len(rewbuffer) > 0:
                        current_reward = statistics.mean(rewbuffer)
                        if current_reward > best_reward:
                            best_reward = current_reward
                            self.save(os.path.join(self.log_dir, f"model_best.pt"), remove_extras=False)
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                import rsl_rl
                git_file_paths = store_code_state(self.log_dir, [rsl_rl.__file__])
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
