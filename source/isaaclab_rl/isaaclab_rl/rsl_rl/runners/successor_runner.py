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
    # Logging — every scalar is keyed on cumulative env_step count.
    # ------------------------------------------------------------------

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Full replacement for BaseRunner.log. Everything is written against
        the cumulative env-step count (BFM-Zero convention), not the iteration
        index. With ``num_steps_per_env=1`` the iteration axis advances much
        faster than env_steps, which makes it useless as a training-progress
        reference; keeping a single env-step axis across all metrics lets
        curves stay comparable across ``num_envs`` / world-size settings.
        """
        import statistics as _stats

        # Counters are advanced in ``learn()`` before log() is called, so
        # here we just read the current total.
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        iteration_time = locs["collection_time"] + locs["learn_time"]
        env_steps = int(self.tot_timesteps)

        # Book-kept scalars used in the terminal printout.
        if hasattr(self.alg.policy, "action_std"):
            mean_std = self.alg.policy.action_std
        else:
            mean_std = 0.0
        if isinstance(mean_std, torch.Tensor):
            mean_std = mean_std.mean().item()
        fps = int(collection_size / max(locs["collection_time"] + locs["learn_time"], 1e-6))

        # ---- Tensorboard / wandb writes — ALL keyed on env_steps ----
        if self.writer is not None and not self.disable_logs:
            # Episode infos (from env)
            ep_string = ""
            if locs.get("ep_infos"):
                for key in locs["ep_infos"][0]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in locs["ep_infos"]:
                        if key not in ep_info:
                            continue
                        v = ep_info[key]
                        if not isinstance(v, torch.Tensor):
                            v = torch.tensor([v])
                        if v.ndim == 0:
                            v = v.unsqueeze(0)
                        infotensor = torch.cat((infotensor, v.to(self.device)))
                    value = torch.mean(infotensor)
                    tag = key if "/" in key else f"Episode/{key}"
                    self.writer.add_scalar(tag, value, env_steps)
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

            # All algorithm-reported scalars — Loss/, Scale/, Critic/, QueryTau/,
            # QueryKeypoint/, Disc/, Style/, Aux/, Action/, Replay/, Reset/, Relabel/
            # — come through in the loss_dict. Each uses its own tag verbatim.
            for tag, value in locs["loss_dict"].items():
                self.writer.add_scalar(tag, value, env_steps)

            # Learning-rate, policy-level telemetry, performance
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, env_steps)
            self.writer.add_scalar("Policy/mean_noise_std", mean_std, env_steps)
            self.writer.add_scalar("Perf/total_fps", fps, env_steps)
            self.writer.add_scalar("Perf/collection_time", locs["collection_time"], env_steps)
            self.writer.add_scalar("Perf/learning_time", locs["learn_time"], env_steps)
            self.writer.add_scalar("Perf/env_steps_total", env_steps, env_steps)
            self.writer.add_scalar("Perf/iteration", locs["it"], env_steps)

            # Training-progress (per-env-buffer) rewards + episode lengths.
            if len(locs["rewbuffer"]) > 0:
                self.writer.add_scalar("Train/mean_reward", _stats.mean(locs["rewbuffer"]), env_steps)
                self.writer.add_scalar("Train/mean_episode_length", _stats.mean(locs["lenbuffer"]), env_steps)
        else:
            ep_string = ""

        # ---- Terminal printout (iteration number + env_steps) ----
        header = f" \033[1m Iter {locs['it']}/{locs['tot_iter']}  env_steps={env_steps:,} \033[0m "
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s,"""
                f""" learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
            log_string += f"""{'Mean reward:':>{pad}} {_stats.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {_stats.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s,"""
                f""" learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        import time as _time
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total env_steps:':>{pad}} {env_steps:,}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {_time.strftime("%H:%M:%S", _time.gmtime(self.tot_time))}\n"""
        )
        print(log_string)

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

        # Total env transitions collected across all ranks. Updated each
        # iteration. Used to decide whether we're still in the BFM-style
        # warmup phase (random actions, no updates) and also as the x-axis
        # for every metric that's logged — gives a device/rank-independent
        # training-progress axis.
        num_seed_steps = int(getattr(self.alg, "num_seed_steps", 0))
        steps_per_iter_global = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # ``tot_timesteps`` is maintained by BaseRunner.log via +=; we shadow
        # it here as self._env_steps_total so decisions based on the counter
        # are consistent within the learn() loop.
        self._env_steps_total = int(self.tot_timesteps)
        # Initial-eval latch — forces an eval on the first iter where the
        # replay has enough data, so the eval panels have an anchor at
        # near-zero env_steps instead of only showing up at the first
        # interval boundary.
        self._did_initial_eval = False

        # Sample uniform-random actions in the actor's output range for the
        # warmup phase. The actor clamps to [action_low, action_high] anyway
        # so matching that range gives the critic the same input distribution
        # it will see at test time.
        action_low = getattr(self.alg.policy.actor, "action_low", -1.0)
        action_high = getattr(self.alg.policy.actor, "action_high", 1.0)

        for it in range(start_iter, tot_iter):
            start = time.time()

            if hasattr(self.env.unwrapped, "pre_rollout"):
                self.env.unwrapped.pre_rollout()

            # BaseRunner.log() increments self.tot_timesteps at the end of
            # each prior iter, so this stays in sync even across resumes.
            self._env_steps_total = int(self.tot_timesteps)
            warmup_active = self._env_steps_total < num_seed_steps

            # ---- Rollout ----
            with torch.inference_mode():
                self.alg.policy.eval()

                for _ in range(self.num_steps_per_env):
                    if warmup_active:
                        # Random actions in the actor's output range. We still
                        # funnel through alg.act() first so the algorithm gets
                        # to stash obs/priv/constraints into self.transition,
                        # then overwrite the actions in-place.
                        _ = alg.act(obs, privileged_obs, infos=infos)
                        rand_actions = torch.empty(
                            self.env.num_envs, self.env.num_actions, device=self.device,
                        ).uniform_(action_low, action_high)
                        alg.transition.actions = rand_actions.detach()
                        actions = rand_actions.clamp(*self.action_clip_range)
                    else:
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
            # Skip updates while we're still populating the replay with random
            # actions (BFM-Zero's ``num_seed_steps`` behaviour).
            if warmup_active:
                loss_dict = {"Perf/warmup": 1.0}
            else:
                self.alg.policy.train()
                loss_dict = self.alg.update()
                loss_dict["Perf/warmup"] = 0.0

            # Sync the aux-reward running normalizer across ranks. Each
            # rank's update() folded its local mini-batch rewards into the
            # Welford stats; without this merge, r_env_norm (and the aux
            # TD target it feeds) diverges per rank even though gradients
            # are averaged.
            if self.is_distributed and not warmup_active:
                aux_norm = getattr(self.alg.policy, "aux_reward_normalizer", None)
                if aux_norm is not None and hasattr(aux_norm, "sync_across_ranks"):
                    aux_norm.sync_across_ranks(self.device)

            # Reset-source breakdown (group 7 diag) — ask the env for RSI share.
            env_u = self.env.unwrapped
            if hasattr(env_u, "consume_reset_stats"):
                reset_stats = env_u.consume_reset_stats()
                loss_dict.update(reset_stats)

            # ---- BFM-style independent tracking eval ----
            # Live rollout with a frozen constraint set per env. The env +
            # algorithm state are snapshotted before the rollout and restored
            # afterwards so the training rollout is not disturbed. Fires:
            #   (a) the first iter after warmup (anchors the eval panel near
            #       env_steps ≈ num_seed_steps),
            #   (b) every time ``env_steps_total`` crosses an integer multiple
            #       of ``eval_interval_env_steps`` after that.
            #
            # Distributed correctness: the firing decision is taken locally
            # but then all-reduced so every rank enters (and exits) the eval
            # together. Without this, a single-rank drift would desync the
            # next ``alg.update()`` collective and hang the job on
            # ``reduce_gradients``.
            eval_interval = int(getattr(self.alg, "eval_interval_env_steps", 0))
            if eval_interval > 0 and not warmup_active:
                post_env_steps = self._env_steps_total + self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
                prev_bucket = self._env_steps_total // eval_interval
                curr_bucket = post_env_steps // eval_interval
                first_eval = not self._did_initial_eval
                interval_crossed = curr_bucket > prev_bucket
                run_eval_local = first_eval or interval_crossed
                # All-reduce the boolean so every rank reaches the same
                # decision — if ANY rank wants to eval, they ALL eval. Keeps
                # NCCL collectives downstream in lockstep.
                if self.is_distributed:
                    import torch.distributed as dist
                    flag = torch.tensor(
                        [1.0 if run_eval_local else 0.0], device=self.device,
                    )
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    run_eval = bool(flag.item() > 0.5)
                else:
                    run_eval = run_eval_local

                if run_eval:
                    # Put the policy + normalizers in eval mode for the
                    # duration: dropout/layernorm stay deterministic and
                    # EmpiricalNormalization stops updating its running
                    # stats on the eval frames.
                    self.alg.policy.eval()
                    prev_obs_norm_mode = self.obs_normalizer.training
                    prev_priv_norm_mode = self.privileged_obs_normalizer.training
                    self.obs_normalizer.eval()
                    self.privileged_obs_normalizer.eval()
                    try:
                        eval_metrics = self.alg.evaluate_live_tracking(
                            self.env,
                            obs_normalizer=self.obs_normalizer,
                            privileged_obs_normalizer=self.privileged_obs_normalizer,
                            privileged_obs_type=self.privileged_obs_type,
                            action_clip_range=self.action_clip_range,
                        )
                    finally:
                        self.alg.policy.train()
                        if prev_obs_norm_mode:
                            self.obs_normalizer.train()
                        if prev_priv_norm_mode:
                            self.privileged_obs_normalizer.train()

                    # Barrier so stragglers (slower sim init, slower snapshot
                    # on nodes with bigger NUMA transfer cost, etc.) don't
                    # drift into the next collective with different wall
                    # clock positions.
                    if self.is_distributed:
                        import torch.distributed as dist
                        dist.barrier()

                    if eval_metrics:
                        loss_dict.update(eval_metrics)
                        self._did_initial_eval = True
                    else:
                        loss_dict["Eval/no_samples"] = 1.0

            if hasattr(self.env.unwrapped, "post_update"):
                env_loss_dict = self.env.unwrapped.post_update()
                if isinstance(env_loss_dict, dict):
                    loss_dict.update(env_loss_dict)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Advance global counters every iter. ``log()`` is called on
            # every iter (no throttling) so it would also advance these,
            # but keeping the bump here lets the Train/global_* writes
            # below read a consistent value without depending on log()'s
            # side-effects.
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
            self.tot_time += collection_time + learn_time

            # ---- Reduce metrics across distributed ranks ----
            # Keyed on env_steps_total (to match our log()) so multi-rank
            # curves overlay cleanly with single-rank baselines.
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
                    # ``tot_timesteps`` has been bumped for this iter just
                    # above, so log() and this block agree on the same
                    # step index (needed to keep wandb happy).
                    env_steps_for_log = int(self.tot_timesteps)
                    self.writer.add_scalar(
                        "Train/global_mean_reward", (reward_t / count_t).item(), env_steps_for_log
                    )
                    self.writer.add_scalar(
                        "Train/global_mean_episode_length", (ep_len_t / count_t).item(), env_steps_for_log
                    )

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
