# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Vision DAgger runner for BFM terrain distillation.

Inherits from FBCprRunner to reuse:
  * z-context rollout (tracking + random mixture via maybe_update_rollout_context)
  * Terrain RSI from tracking
  * obs_to_device dict conversion
  * Expert buffer loading + sampling

Differences from FBCprRunner:
  * No DDPG/FB-CPR gradient updates — pure DAgger (MSE on teacher actions)
  * Student policy: VisionStudent(depth_img, proprio, z) → action
  * Teacher: the loaded FB-CPR policy's actor(obs_dict, z) → action
  * Rollout uses DAgger beta-mixing (student + teacher)
  * Training applies re-labeling: re-sample z and re-compute teacher actions
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Dict

import torch
import torch.nn as nn

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

from isaaclab_rl.rsl_rl.modules.empirical_normalization import EmpiricalNormalization
from isaaclab_rl.rsl_rl.modules.vision_student import VisionStudent
from isaaclab_rl.rsl_rl.modules.depth_noise import D435iDepthNoise

from .fb_cpr_runner import FBCprRunner


class VisionDAggerRunner(FBCprRunner):
    """DAgger runner: FB-CPR teacher (height-scan) → Vision student (depth).

    The teacher IS the loaded FB-CPR policy (self.policy). The student is an
    additional VisionStudent network that replaces height_scan with depth.

    Z handling: identical to FBCprRunner — tracking + random mixture via
    ``self.alg.maybe_update_rollout_context``. On reset, terrain RSI is
    applied. During training, re-labeling re-samples z from the backward
    map on stored observations and recomputes teacher actions.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cuda:0",
        **kwargs,
    ) -> None:
        # Initialize the full FB-CPR runner (loads teacher policy, expert buffer, etc.)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device, **kwargs)

        # Load teacher checkpoint (the FB-CPR policy weights)
        dagger_cfg = self.alg_cfg.get("dagger", {})
        teacher_ckpt_path = dagger_cfg.get("teacher_ckpt", "")
        if teacher_ckpt_path:
            print(f"[VisionDAggerRunner] Loading teacher from: {teacher_ckpt_path}", flush=True)
            super().load(teacher_ckpt_path, load_optimizer=False)

        # Freeze teacher (the loaded FB-CPR policy)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

        # Build the vision student
        dagger_cfg = self.alg_cfg.get("dagger", {})
        self.depth_height = dagger_cfg.get("depth_height", 58)
        self.depth_width = dagger_cfg.get("depth_width", 87)

        # Proprio = state + last_action + history (everything except height_scan).
        state_dim = self._obs_key_groups["state"]["dim"]
        last_action_dim = self._obs_key_groups.get("last_action", {}).get("dim", 0)
        history_dim = self._obs_key_groups.get("history_actor", {}).get("dim", 0)
        # global_xy_target is fed separately (with 50% dropout)
        self._global_xy_dim = self._obs_key_groups.get("global_xy_target", {}).get("dim", 0)
        self._student_proprio_dim = state_dim + last_action_dim + history_dim + self._global_xy_dim

        self._z_dim = self.policy.cfg.z_dim
        self.student = VisionStudent(
            num_proprio=self._student_proprio_dim,
            num_actions=self.action_dim,
            z_dim=self._z_dim,
            depth_height=self.depth_height,
            depth_width=self.depth_width,
            depth_feature_dim=dagger_cfg.get("depth_feature_dim", 128),
            hidden_dim=dagger_cfg.get("hidden_dim", 2048),
            hidden_layers=dagger_cfg.get("hidden_layers", 6),
            embedding_layers=dagger_cfg.get("embedding_layers", 2),
        ).to(self.device)

        # Student optimizer
        self._dagger_lr = dagger_cfg.get("learning_rate", 3e-4)
        self._dagger_weight_decay = dagger_cfg.get("weight_decay", 1e-4)
        self._dagger_epochs = dagger_cfg.get("num_learning_epochs", 4)
        self._dagger_max_grad_norm = dagger_cfg.get("max_grad_norm", 1.0)
        self._dagger_batch_size = dagger_cfg.get("batch_size", 2048)
        self._dagger_relabel_ratio = dagger_cfg.get("relabel_ratio", 0.5)

        # Wrap student in DDP if distributed
        if self.is_distributed:
            torch.distributed.barrier()
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.student = DDP(
                self.student,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        self._student_module = self.student.module if self.is_distributed else self.student

        self.student_optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self._dagger_lr,
            weight_decay=self._dagger_weight_decay,
        )

        # Proprio normalizer (separate from teacher's obs normalizer)
        self.proprio_normalizer = EmpiricalNormalization(
            shape=[self._student_proprio_dim], until=1.0e8
        ).to(self.device)

        # D435i depth noise model for realistic training augmentation
        self._depth_noise = D435iDepthNoise(
            z_min=0.3,
            z_max=dagger_cfg.get("depth_max_dist", 3.0),
            alpha=dagger_cfg.get("depth_noise_alpha", 0.005),
            beta=dagger_cfg.get("depth_noise_beta", 0.001),
        )

        # Rollout buffer for DAgger training
        self._dagger_buffer_depth: list[torch.Tensor] = []
        self._dagger_buffer_proprio: list[torch.Tensor] = []
        self._dagger_buffer_z: list[torch.Tensor] = []
        self._dagger_buffer_teacher_obs: list[dict[str, torch.Tensor]] = []
        self._dagger_buffer_teacher_action: list[torch.Tensor] = []


    def _get_student_proprio(self, obs_dict: dict[str, torch.Tensor], dropout_global: bool = True) -> torch.Tensor:
        """Extract proprio for student (state + last_action + history + global_xy_target).

        global_xy_target is included with 50% per-env dropout (zeroed out)
        to match training-time behavior of the Global FB env.
        """
        parts = [obs_dict["state"]]
        if "last_action" in obs_dict:
            parts.append(obs_dict["last_action"])
        if "history_actor" in obs_dict:
            parts.append(obs_dict["history_actor"])
        if "global_xy_target" in obs_dict and self._global_xy_dim > 0:
            gxy = obs_dict["global_xy_target"]
            if dropout_global and self.student.training:
                mask = (torch.rand(gxy.shape[0], 1, device=gxy.device) > 0.5).float()
                gxy = gxy * mask
            parts.append(gxy)
        elif self._global_xy_dim > 0:
            parts.append(torch.zeros(obs_dict["state"].shape[0], self._global_xy_dim, device=self.device))
        return torch.cat(parts, dim=-1)

    def _get_depth_image(self) -> torch.Tensor:
        """Read depth image from the env's depth sensor.

        When using RTX camera, noise + range gating is handled inside
        the env's compute_rtx_depth(). For pseudo-depth, the runner
        applies D435i noise.
        Returns: [N, H, W] tensor.
        """
        eu = self.env_unwrapped
        depth_cam = eu.depth_camera
        depth = depth_cam.data.output["depth"].squeeze(-1)  # [N, H, W]
        max_dist = getattr(eu.cfg, "depth_camera_max_distance", 3.0)
        depth[torch.isinf(depth)] = 0.0
        depth[torch.isnan(depth)] = 0.0
        depth = depth.clamp(0.0, max_dist)
        # Apply noise only for pseudo-depth (RTX path handles noise internally)
        if self.student.training and not getattr(eu.cfg, "depth_use_rtx", False):
            depth = self._depth_noise(depth)
        return depth.to(self.device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        self._is_head = (not self.is_distributed) or (self.gpu_global_rank == 0)

        # DDP: broadcast teacher weights from rank 0 so all ranks are aligned.
        if self.is_distributed:
            self.alg.broadcast_parameters()

        # Setup logging
        if self.log_dir is not None and self.writer is None and self._is_head:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            if self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            else:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # DDP sync is handled by the DDP wrapper automatically

        # Initial obs
        obs_flat, extras = self.env.get_observations()
        obs_dict = self._obs_to_device(obs_flat, extras)
        step_count = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.device)

        # Initialize z-context (same as FBCprRunner)
        _robot = self.env_unwrapped.robot if hasattr(self.env_unwrapped, "robot") else None
        _terrain_z_fn = getattr(self.env_unwrapped, "_get_terrain_height_xy", None)
        z_context, terrain_reset = self.alg.maybe_update_rollout_context(
            z=None, step_count=step_count, expert_buffer=self.expert_buffer,
            robot_root_xy=_robot.data.root_pos_w[:, :2].to(self.device) if _robot else None,
            robot_root_quat=_robot.data.root_quat_w.to(self.device) if _robot else None,
            terrain_z_fn=_terrain_z_fn,
        )
        if terrain_reset is not None:
            env_ids = terrain_reset["env_ids"]
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

        rewbuffer: deque[float] = deque(maxlen=500)
        lenbuffer: deque[float] = deque(maxlen=500)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_ep_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()
            self._dagger_buffer_depth.clear()
            self._dagger_buffer_proprio.clear()
            self._dagger_buffer_z.clear()
            self._dagger_buffer_teacher_obs.clear()
            self._dagger_buffer_teacher_action.clear()

            # ---- Rollout (DAgger) ----
            self.student.eval()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Get depth image
                    depth = self._get_depth_image()
                    proprio = self._get_student_proprio(obs_dict)

                    # Teacher labels (uses full obs_dict including height_scan)
                    teacher_action = self.policy.act(obs_dict, z_context, mean=True)

                    # Student executes (pure DAgger — no beta mixing)
                    rollout_action = self._student_module.act_inference(
                        depth, self.proprio_normalizer(proprio), z_context,
                    )

                    # Store for training
                    self._dagger_buffer_depth.append(depth)
                    self._dagger_buffer_proprio.append(proprio)
                    self._dagger_buffer_z.append(z_context.clone())
                    self._dagger_buffer_teacher_obs.append(
                        {k: v.clone() for k, v in obs_dict.items()}
                    )
                    self._dagger_buffer_teacher_action.append(teacher_action)

                    # Push global FB targets to env
                    env_u = self.env_unwrapped
                    if hasattr(env_u, "set_global_fb_targets"):
                        targets = self.alg.get_global_fb_targets(step_count, self.expert_buffer)
                        if targets is not None:
                            env_u.set_global_fb_targets(*targets)

                    # Step env
                    new_obs, rewards, dones, infos = self.env.step(rollout_action.to(self.env.device))
                    new_obs = self._obs_to_device(new_obs, infos)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Book-keeping
                    cur_reward_sum += rewards
                    cur_ep_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_ep_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_ep_length[new_ids] = 0

                    # Update step count
                    step_count = step_count + 1
                    step_count = torch.where(dones.bool(), torch.zeros_like(step_count), step_count)

                    # Update z context (tracking + random, same as FBCprRunner)
                    z_context, terrain_reset = self.alg.maybe_update_rollout_context(
                        z=z_context, step_count=step_count, expert_buffer=self.expert_buffer,
                        robot_root_xy=_robot.data.root_pos_w[:, :2].to(self.device) if _robot else None,
                        robot_root_quat=_robot.data.root_quat_w.to(self.device) if _robot else None,
                        terrain_z_fn=_terrain_z_fn,
                    )

                    # Terrain variant root_h flag sync
                    grh = getattr(self.alg, "_tracking_terrain_variant_root_h", None)
                    if grh is not None and hasattr(self.env_unwrapped, "_use_terrain_variant_root_h"):
                        self.env_unwrapped._use_terrain_variant_root_h = grh

                    # Terrain RSI on tracking reset
                    if terrain_reset is not None and hasattr(self.env_unwrapped, "_reset_idx"):
                        env_ids = terrain_reset["env_ids"]
                        already_done = dones[env_ids].bool()
                        if not already_done.all():
                            mask = ~already_done
                            need_reset = env_ids[mask]
                            self._terrain_rsi_from_tracking(
                                need_reset,
                                terrain_reset["motion_ids"][mask],
                                terrain_reset["starts"][mask],
                            )
                            step_count[need_reset] = 0
                            dones[need_reset] = 1
                            fresh_obs, fresh_extras = self.env.get_observations()
                            new_obs = self._obs_to_device(fresh_obs, fresh_extras)
                            if _robot is not None:
                                self.alg.update_tracking_pose_after_reset(
                                    need_reset,
                                    _robot.data.root_pos_w[:, :2].to(self.device),
                                    _robot.data.root_quat_w.to(self.device),
                                )

                    obs_dict = new_obs

            collection_time = time.time() - start
            start = time.time()

            # ---- Train student (DAgger with re-labeling) ----
            self.student.train()
            self.proprio_normalizer.train()

            # Stack buffer into flat tensors
            depth_all = torch.cat(self._dagger_buffer_depth, dim=0)       # [N*T, H, W]
            proprio_all = torch.cat(self._dagger_buffer_proprio, dim=0)   # [N*T, D]
            z_all = torch.cat(self._dagger_buffer_z, dim=0)               # [N*T, z_dim]
            target_all = torch.cat(self._dagger_buffer_teacher_action, dim=0)  # [N*T, A]

            # Re-labeling: same as FB-CPR's sample_mixed_z — mix of
            # backward_map(shuffled obs), expert-encoded z, and random z.
            # Recompute teacher action under the new z.
            total_samples = depth_all.shape[0]
            n_relabel = int(total_samples * self._dagger_relabel_ratio)
            if n_relabel > 0:
                relabel_idx = torch.randperm(total_samples, device=self.device)[:n_relabel]
                teacher_obs_all = {
                    k: torch.cat([d[k] for d in self._dagger_buffer_teacher_obs], dim=0)
                    for k in self._dagger_buffer_teacher_obs[0].keys()
                }
                relabel_obs = {k: v[relabel_idx] for k, v in teacher_obs_all.items()}

                with torch.inference_mode():
                    # Mixed z: 1/3 backward_map(shuffled), 1/3 expert, 1/3 random
                    n = n_relabel
                    n_third = n // 3

                    # Backward map on shuffled obs
                    perm = torch.randperm(n, device=self.device)
                    shuffled_obs = {k: v[perm] for k, v in relabel_obs.items()}
                    z_goal = self.policy.backward_map(shuffled_obs)
                    z_goal = self.policy.project_z(z_goal)

                    # Expert-encoded z — sample n_third transitions, encode via B
                    expert_chunks = self.expert_buffer.sample_chunks(
                        n_third, 1, target_device=self.device,
                    )
                    expert_batch = next(iter(expert_chunks))
                    expert_next_obs = expert_batch["next"]["observation"]
                    B_out = self.policy._backward_map(expert_next_obs).detach()
                    z_expert = self.policy.project_z(B_out)

                    # Random z
                    z_random = self.policy.sample_z(n, device=self.device)

                    # Mix
                    new_z = z_random.clone()
                    new_z[:n_third] = z_goal[:n_third]
                    new_z[n_third:2*n_third] = z_expert[:n_third]

                    # Recompute teacher actions
                    new_teacher_action = self.policy.act(relabel_obs, new_z, mean=True)

                z_all[relabel_idx] = new_z
                target_all[relabel_idx] = new_teacher_action

            # Mini-batch SGD (DDP handles gradient sync automatically)
            mean_loss = 0.0
            num_batches = 0
            for _ in range(self._dagger_epochs):
                perm = torch.randperm(total_samples, device=self.device)
                for start_idx in range(0, total_samples, self._dagger_batch_size):
                    idx = perm[start_idx:start_idx + self._dagger_batch_size]
                    d_batch = depth_all[idx]
                    p_batch = self.proprio_normalizer(proprio_all[idx])
                    z_batch = z_all[idx]
                    t_batch = target_all[idx]

                    pred = self.student(d_batch, p_batch, z_batch)
                    loss = nn.functional.mse_loss(pred, t_batch)

                    self.student_optimizer.zero_grad()
                    loss.backward()
                    if self._dagger_max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self._student_module.parameters(), self._dagger_max_grad_norm
                        )
                    self.student_optimizer.step()
                    mean_loss += loss.item()
                    num_batches += 1

            mean_loss /= max(num_batches, 1)
            learn_time = time.time() - start
            self.current_learning_iteration = it
            self.tot_timesteps += self.env.num_envs * self.num_steps_per_env * self.gpu_world_size
            self.tot_time += collection_time + learn_time

            # ---- Logging ----
            if self.log_dir is not None and self._is_head:
                fps = int(self.env.num_envs * self.num_steps_per_env * self.gpu_world_size
                          / (collection_time + learn_time))
                if self.writer:
                    self.writer.add_scalar("Loss/dagger_mse", mean_loss, it)
                    self.writer.add_scalar("Perf/total_fps", fps, it)
                    self.writer.add_scalar("Perf/collection_time", collection_time, it)
                    self.writer.add_scalar("Perf/learn_time", learn_time, it)
                    if rewbuffer:
                        self.writer.add_scalar("Train/mean_reward", sum(rewbuffer) / len(rewbuffer), it)
                    if lenbuffer:
                        self.writer.add_scalar("Train/mean_ep_length", sum(lenbuffer) / len(lenbuffer), it)

                print(
                    f"[{it}/{tot_iter}] loss={mean_loss:.5f} "
                    f"fps={fps} rew={sum(rewbuffer)/max(len(rewbuffer),1):.2f} "
                    f"coll={collection_time:.2f}s learn={learn_time:.2f}s",
                    flush=True,
                )

                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            if it == start_iter and self._is_head:
                git_file_paths = store_code_state(self.log_dir, getattr(self, "git_status_repos", []))
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and self._is_head:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos=None):
        torch.save({
            "student_state_dict": self._student_module.state_dict(),
            "optimizer_state_dict": self.student_optimizer.state_dict(),
            "proprio_normalizer_state_dict": self.proprio_normalizer.state_dict(),
            "iter": self.current_learning_iteration,
            "student_cfg": {
                "num_proprio": self._student_module.num_proprio,
                "num_actions": self._student_module.num_actions,
                "z_dim": self._student_module.z_dim,
                "depth_height": self._student_module.depth_height,
                "depth_width": self._student_module.depth_width,
            },
            "teacher_ckpt": self.cfg.get("resume_path", ""),
        }, path)

    def load(self, path: str, load_optimizer: bool = True):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "student_state_dict" in ckpt:
            self._student_module.load_state_dict(ckpt["student_state_dict"])
            if "proprio_normalizer_state_dict" in ckpt:
                self.proprio_normalizer.load_state_dict(ckpt["proprio_normalizer_state_dict"])
            if load_optimizer and "optimizer_state_dict" in ckpt:
                self.student_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "iter" in ckpt:
                self.current_learning_iteration = ckpt["iter"]
        else:
            # Loading a teacher checkpoint — handled by parent
            super().load(path, load_optimizer=False)
        return ckpt

    def get_inference_policy(self, device=None):
        """Return student inference function for deployment."""
        self._student_module.eval()
        if device:
            self._student_module.to(device)
            self.proprio_normalizer.to(device)
        normalizer = self.proprio_normalizer
        student = self._student_module

        def policy_fn(depth, proprio, z):
            return student.act_inference(depth, normalizer(proprio), z)

        return policy_fn
