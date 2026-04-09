# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import store_code_state

from isaaclab_rl.rsl_rl.algorithms import *
from isaaclab_rl.rsl_rl.modules import *
from isaaclab_rl.rsl_rl.utils import broadcast_parameters, reduce_gradients

from collections import deque
from copy import deepcopy

def _sync_normalizer(normalizer, device):
    """All-reduce empirical normalizer stats across distributed ranks.

    Combines running mean/var/count from all ranks using parallel statistics merging
    so each rank benefits from observations seen by every other rank.
    """
    if not isinstance(normalizer, EmpiricalNormalization):
        return
    import torch.distributed as dist

    count = normalizer.count.clone().float().to(device)
    mean = normalizer._mean.clone().to(device)
    var = normalizer._var.clone().to(device)

    # Gather counts from all ranks to compute proper weighted merge
    world_size = dist.get_world_size()
    all_counts = [torch.zeros_like(count) for _ in range(world_size)]
    all_means = [torch.zeros_like(mean) for _ in range(world_size)]
    all_vars = [torch.zeros_like(var) for _ in range(world_size)]

    dist.all_gather(all_counts, count)
    dist.all_gather(all_means, mean)
    dist.all_gather(all_vars, var)

    # Merge using parallel Welford's algorithm
    total_count = all_counts[0]
    merged_mean = all_means[0]
    merged_var = all_vars[0]

    for i in range(1, world_size):
        n_a = total_count
        n_b = all_counts[i]
        n_ab = n_a + n_b
        if n_ab < 1:
            continue
        delta = all_means[i] - merged_mean
        merged_var = (n_a * merged_var + n_b * all_vars[i] + delta.pow(2) * n_a * n_b / n_ab) / n_ab
        merged_mean = (n_a * merged_mean + n_b * all_means[i]) / n_ab
        total_count = n_ab

    normalizer._mean.data.copy_(merged_mean)
    normalizer._var.data.copy_(merged_var)
    normalizer._std.data.copy_(torch.sqrt(merged_var))
    normalizer.count.data.copy_(total_count.long())

class BaseRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", **kwargs):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.env_unwrapped = env.unwrapped # type: ignore
        if 'eval_mode' in kwargs:
            self.load_actor_only = kwargs['eval_mode']
        else:
            self.load_actor_only = False

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # check if checkpoint should be uploaded
        self.upload_checkpoint = self.cfg.get("upload_checkpoint", True)

        # max checkpoint number
        self.max_checkpoint_num = self.cfg.get("max_checkpoint_num", 10)

        # action clip range
        self.action_clip_range = self.cfg.get("action_clip_range", [-50.0, 50.0])

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] in ["Distillation", "FlowDAgger"]:
            self.training_type = "distillation"
        else:
            print(f"Warning: Training type not found for algorithm {self.alg_cfg['class_name']}, using rl as default.")
            self.training_type = "rl"

        # resolve dimensions of observations
        obs_dict = self.env.unwrapped._get_observations(compute_meta=True)
        meta_tensors = self.env.unwrapped._get_meta_tensors()
        if len(meta_tensors) == 0:
            meta_tensors = None
        else:
            print(f"[INFO]: Meta tensors are used, keys: {meta_tensors.keys()}")
        num_obs = obs_dict['policy'].shape[1]

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in obs_dict:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None
            meta_dict = dict(
                actor_obs_meta=obs_dict['policy_meta'],
                critic_obs_meta=obs_dict['critic_meta'],
            )
        if self.training_type == "distillation":
            if "teacher" in obs_dict:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None
            meta_dict = dict(
                student_obs_meta=obs_dict['policy_meta'],
                teacher_obs_meta=obs_dict['teacher_meta'],
            )

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = obs_dict[self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        if self.training_type == "distillation" and hasattr(self.env_unwrapped, "_get_main_observations"): 
            obs = self.env_unwrapped._get_main_observations()
            num_student_priv_obs = obs['critic'].shape[1]
            num_obs = (num_obs, num_student_priv_obs)
            print(f"[INFO]: Student privileged observations shape: (*, {num_student_priv_obs})")

        self.full_policy_cfg = deepcopy(self.policy_cfg)
        self.full_policy_cfg["_args"] = [num_obs, num_privileged_obs, self.env.num_actions]
        self.full_policy_cfg.update(meta_dict)

        # extract distributed training flags from policy config and forward to multi_gpu_cfg
        distributed_critic = self.policy_cfg.pop("distributed_critic", False)
        distributed_actor = self.policy_cfg.pop("distributed_actor", False)
        self.distributed_s3_prefix = self.policy_cfg.pop("distributed_s3_prefix", "")
        if distributed_actor:
            distributed_critic = True  # distributed_actor implies distributed_critic
        if distributed_critic and self.multi_gpu_cfg is not None:
            self.multi_gpu_cfg["distributed_critic"] = True
            self.multi_gpu_cfg["distributed_actor"] = distributed_actor
            if distributed_actor:
                print(f"[INFO]: Fully distributed training — no gradient sync. Each rank trains independently.")
            else:
                print(f"[INFO]: Distributed critic enabled — each rank's critic will not sync gradients.")

        # For distributed distillation (MoE→student): each rank loads its own expert as teacher
        if "teacher_policy_ckpt" in self.policy_cfg and "{rank}" in str(self.policy_cfg["teacher_policy_ckpt"]):
            self.policy_cfg["teacher_policy_ckpt"] = self.policy_cfg["teacher_policy_ckpt"].replace(
                "{rank}", str(self.gpu_global_rank)
            )
            print(f"[INFO]: Rank {self.gpu_global_rank} loading teacher from: {self.policy_cfg['teacher_policy_ckpt']}")

        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg, **meta_dict
        ).to(self.device)

        if isinstance(num_obs, tuple):
            num_obs, _ = num_obs

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = obs_dict.get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= self.env_unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        self.privileged_empirical_normalization_only = self.cfg.get("privileged_empirical_normalization_only", False)
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        if self.training_type == "distillation" and policy.obs_norm_state_dict is not None:
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8)
            self.privileged_obs_normalizer.load_state_dict(policy.obs_norm_state_dict)
            self.privileged_obs_normalizer = self.privileged_obs_normalizer.to(self.device).eval()
            policy.obs_norm_state_dict = None # type: ignore
            print('[INFO]: Loaded teacher empirical normalizer')

        elif self.training_type == "distillation" and policy.obs_norm_state_dict is None:
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)
            print('[INFO]: No teacher empirical normalizer loaded')

        if self.privileged_empirical_normalization_only:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            print('[INFO]: Only privileged observations are normalized')

        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
            meta_tensors=meta_tensors
        )

        # init AMP reward
        if "amp_cfg" in self.cfg and self.cfg["amp_cfg"] is not None and not self.env_unwrapped.cfg.play_mode:
            # add AMP to observation space
            amp_obs = obs_dict["amp_policy"]
            amp_dict: dict[str, torch.Tensor] = {}
            if isinstance(amp_obs, torch.Tensor):
                amp_dict[''] = amp_obs
            elif isinstance(amp_obs, dict):
                amp_dict.update(amp_obs)
            else:
                raise ValueError(f"AMP observations must be a tensor or a dictionary, got {type(amp_obs)}")

            self.cfg["amp_cfg"].pop("input_dim")
            reward_scale = self.cfg["amp_cfg"].pop("reward_scale")
            reward_exp = self.cfg["amp_cfg"].pop("reward_exp")
            w_grad_penalty = self.cfg["amp_cfg"].pop("w_grad_penalty")
            if isinstance(reward_scale, float):
                reward_scale = {k: reward_scale for k in amp_dict.keys()}
            if isinstance(reward_exp, float):
                reward_exp = {k: reward_exp for k in amp_dict.keys()}
            if isinstance(w_grad_penalty, float):
                w_grad_penalty = {k: w_grad_penalty for k in amp_dict.keys()}
            self.amp_rewards = {k: AmpReward(v.shape[1], training=True, 
                                        num_envs=self.env.num_envs,
                                        device=self.device, 
                                        multi_gpu_cfg=self.multi_gpu_cfg,
                                        reward_scale=reward_scale[k],
                                        reward_exp=reward_exp[k],
                                        w_grad_penalty=w_grad_penalty[k],
                                        **self.cfg["amp_cfg"]) for k, v in amp_dict.items()}
        else:
            self.amp_rewards = None

        # init SMP reward (Score-Matching Motion Prior)
        if "smp_cfg" in self.cfg and self.cfg["smp_cfg"] is not None and not self.env_unwrapped.cfg.play_mode:
            from latentctrl.tasks.direct.motion_imitation.smp_reward import SmpReward
            smp_cfg = self.cfg["smp_cfg"].copy()
            self.smp_reward = SmpReward(
                num_envs=self.env.num_envs,
                device=self.device,
                **smp_cfg,
            )
            self.smp_reward.eval()
            print(f"[INFO]: SMP reward initialized with config: {smp_cfg}")
        else:
            self.smp_reward = None

        # Decide whether to disable logging
        # With distributed_actor, every rank logs and saves independently
        self.distributed_actor = distributed_actor and self.is_distributed
        if self.distributed_actor:
            self.disable_logs = False
        else:
            self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging — each rank gets its own subdirectory when distributed_actor is enabled.
        # Sync log_dir from rank 0 so all ranks share the same base dir (and wandb group).
        if self.distributed_actor and log_dir is not None:
            import torch.distributed as dist
            import time as _time
            # Broadcast rank 0's timestamp so all ranks use the same log directory
            # log_dir ends with a timestamp like "2026-04-02_00-01-31[_run_name]"
            # We broadcast rank 0's epoch time and reconstruct the same string on all ranks
            ts = torch.tensor([_time.time()], dtype=torch.double, device=self.device)
            dist.broadcast(ts, src=0)
            from datetime import datetime
            synced_ts = datetime.fromtimestamp(ts.item()).strftime("%Y-%m-%d_%H-%M-%S")
            # Replace this rank's timestamp with rank 0's in the log_dir
            base = os.path.dirname(log_dir)  # .../logs/rsl_rl/experiment_name
            dir_name = os.path.basename(log_dir)  # timestamp[_run_name]
            # The timestamp is always the first 19 chars (YYYY-MM-DD_HH-MM-SS)
            suffix = dir_name[19:]  # e.g. "_run_name" or ""
            log_dir = os.path.join(base, synced_ts + suffix)
            self.log_dir = os.path.join(log_dir, f"rank_{self.gpu_global_rank}")
        else:
            self.log_dir = log_dir
        # Collect per-rank expert metadata (e.g. assigned motion names) for distributed training
        self.expert_metas = self._collect_expert_metas() if self.distributed_actor else None
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                # MoE: each rank gets its own grouped wandb run so every rank logs directly
                if self.distributed_actor:
                    import wandb
                    run_name = os.path.split(self.log_dir)[-1]
                    # parent dir name is the group (shared experiment timestamp)
                    group_name = os.path.basename(os.path.dirname(self.log_dir))
                    project = self.cfg.get("wandb_project", "LatentControl")
                    entity = os.environ.get("WANDB_USERNAME", None)
                    wandb.init(
                        project=project, entity=entity,
                        name=f"expert_{self.gpu_global_rank}",
                        group=group_name,
                        job_type=f"rank_{self.gpu_global_rank}",
                    )
                    wandb.config.update({"log_dir": self.log_dir, "rank": self.gpu_global_rank})
                    if self.expert_metas:
                        wandb.config.update({"expert_metas": self.expert_metas})
                    # Patch wandb.init to no-op so WandbSummaryWriter doesn't create a second run
                    _orig_wandb_init = wandb.init
                    wandb.init = lambda *a, **kw: wandb.run
                    try:
                        self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                    finally:
                        wandb.init = _orig_wandb_init
                else:
                    self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter # type: ignore

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)
        if self.training_type == "distillation":
            self.privileged_obs_normalizer.eval() # no updates for teacher normalizer

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=500)
        lenbuffer = deque(maxlen=500)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=500)
            irewbuffer = deque(maxlen=500)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # Sync empirical normalizers across ranks (critical for MoE where each rank sees different data)
            if self.empirical_normalization:
                _sync_normalizer(self.obs_normalizer, self.device)
                _sync_normalizer(self.privileged_obs_normalizer, self.device)
            if self.amp_rewards is not None:
                for k in self.amp_rewards.keys():
                    broadcast_parameters(self.amp_rewards[k].network)

        if self.amp_rewards is not None:
            amp_reward_storages = {k: torch.zeros(self.env.num_envs, device=self.device) for k in self.amp_rewards.keys()}
            for k in self.amp_rewards.keys():
                self.amp_rewards[k].reset_storage()

        # SMP trajectory buffers
        if self.smp_reward is not None:
            smp_body_pos_list = []
            smp_body_quat_list = []
            smp_body_vel_list = []
            smp_foot_contact_list = []
            smp_reward_storage = torch.zeros(self.env.num_envs, device=self.device)
            # Resolve foot contact body indices once
            contact_sensor = self.env_unwrapped.robot_contact_sensor
            smp_foot_body_ids = [
                contact_sensor.body_names.index(n)
                for n in ['left_ankle_roll_link', 'right_ankle_roll_link']
            ]

        # initialize infos
        infos = dict(meta_tensors={})

        # initialize best reward
        best_reward = -float("inf")

        # set unwrapped env to algorithm
        if not hasattr(self.alg, "unwrapped_env"):
            setattr(self.alg, "unwrapped_env", self.env.unwrapped) # type: ignore
        else:
            print(f"[WARNING]: Unwrapped env already set to algorithm, skipping...")

        # sync multi gpu
        if hasattr(self.env.unwrapped, "sync_multi_gpu"):
            self.env.unwrapped.sync_multi_gpu(self.multi_gpu_cfg)

        # Actor warmup for finetuning: use small actor lr while critic warms up
        self.actor_freeze_iters = self.cfg.get("actor_freeze_iterations", 0)
        self.actor_warmup_lr_scale = self.cfg.get("actor_warmup_lr_scale", 0.1)
        if self.actor_freeze_iters > 0:
            self._actor_lr_backup = self.alg.optimizer.param_groups[0]["lr"]
            self.alg.optimizer.param_groups[0]["lr"] = self._actor_lr_backup * self.actor_warmup_lr_scale
            if len(self.alg.optimizer.param_groups) > 2:
                self._other_lr_backup = self.alg.optimizer.param_groups[2]["lr"]
                self.alg.optimizer.param_groups[2]["lr"] = self._other_lr_backup * self.actor_warmup_lr_scale
            print(f"[INFO]: Actor warmup for first {self.actor_freeze_iters} iterations "
                  f"(lr_scale={self.actor_warmup_lr_scale})")

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # call pre_rollout method of the environment
            if hasattr(self.env.unwrapped, "pre_rollout"):
                self.env.unwrapped.pre_rollout()

            # Rollout
            with torch.inference_mode():
                self.alg.policy.eval()
                if getattr(self.alg, "collect_reset", False):
                    self.alg.policy.reset() # type: ignore

                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs, infos=infos).clamp(*self.action_clip_range)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device)) # type: ignore
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs
                    # try to get all resets
                    resets = infos.get("resets", dones == 1).to(self.device)
                    if getattr(self.alg, "done_reset", False) and hasattr(self.alg.policy, "reset"):  
                        self.alg.policy.reset(resets) # type: ignore

                    if self.amp_rewards is not None:
                        reward_scale = infos.get("overall_reward_scale", 1.0)
                        gen_obs = infos["observations"]["amp_policy"]
                        ref_obs = infos["observations"]["amp_motion"]
                        if isinstance(gen_obs, torch.Tensor):
                            gen_obs = {'': gen_obs.to(self.device)}
                        if isinstance(ref_obs, torch.Tensor):
                            ref_obs = {'': ref_obs.to(self.device)}
                        for k, v in gen_obs.items():
                            self.amp_rewards[k].update_storage(v, ref_obs[k])

                        amp_reward_scale = 1.0
                        if 'reward_scales' in infos and 'amp' in infos['reward_scales']:
                            amp_reward_scale = infos['reward_scales']['amp']
                            if isinstance(amp_reward_scale, torch.Tensor):
                                amp_reward_scale = amp_reward_scale.to(self.device)

                        for k in self.amp_rewards.keys():
                            amp_reward = self.amp_rewards[k].compute_reward(gen_obs[k], amp_reward_scale) * reward_scale
                            amp_reward_storages[k] += amp_reward
                            rewards += amp_reward

                            # update episode info
                            key_name = f"rew_amp_{k}" if k != '' else 'rew_amp'
                            infos["episode"][key_name] = amp_reward_storages[k].mean().item()
                            infos["episode"]["Perstep/" + key_name] = (amp_reward_storages[k] / infos["episode_length"].to(self.device)).mean().item()
                            infos["episode"]["Discriminator/" + key_name + "/mean"] = amp_reward.mean().item()
                            infos["episode"]["Discriminator/" + key_name + "/std"] = amp_reward.std().item()
                            infos["episode"]["Discriminator/" + key_name + "/min"] = amp_reward.min().item()
                            infos["episode"]["Discriminator/" + key_name + "/max"] = amp_reward.max().item()
                            amp_reward_storages[k][dones == 1] = 0.

                            # update total reward and perstep total reward
                            infos["episode"]["total_reward"] += infos["episode"][key_name]
                            infos["episode"]["Perstep/total_reward"] += infos["episode"]["Perstep/" + key_name]

                    # collect body states for SMP trajectory reward
                    if self.smp_reward is not None:
                        smp_body_pos_list.append(self.env_unwrapped.body_pos.clone())
                        smp_body_quat_list.append(self.env_unwrapped.body_quat.clone())
                        smp_body_vel_list.append(self.env_unwrapped.body_lin_vel.clone())
                        # Foot contact: force norm > 1N → contact=1
                        foot_forces = contact_sensor.data.net_forces_w[:, smp_foot_body_ids]  # [N, 2, 3]
                        foot_in_contact = (foot_forces.norm(dim=-1) > 1.0).float()  # [N, 2]
                        smp_foot_contact_list.append(foot_in_contact)

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None # type: ignore

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                # Compute trajectory-level SMP reward and add to rollout storage
                if self.smp_reward is not None:
                    smp_body_pos_seq = torch.stack(smp_body_pos_list, dim=0)       # [T, N, J, 3]
                    smp_body_quat_seq = torch.stack(smp_body_quat_list, dim=0)     # [T, N, J, 4]
                    smp_body_vel_seq = torch.stack(smp_body_vel_list, dim=0)       # [T, N, J, 3]
                    smp_foot_contact_seq = torch.stack(smp_foot_contact_list, dim=0)  # [T, N, 2]
                    smp_rewards, smp_raw_reward = self.smp_reward.compute_trajectory_reward(
                        smp_body_pos_seq, smp_body_quat_seq,
                        body_lin_vel_seq=smp_body_vel_seq,
                        foot_contacts_seq=smp_foot_contact_seq,
                    )  # [T, N, 1], [N, T]
                    # Add SMP reward to stored rewards retroactively
                    self.alg.storage.rewards[:self.num_steps_per_env] += smp_rewards.to(self.device)

                    # Log per-step SMP stats (these are per-rollout, not per-episode)
                    if "episode" in infos:
                        infos["episode"]["SMP/mean"] = smp_rewards.mean().item()
                        infos["episode"]["SMP/std"] = smp_rewards.std().item()
                        infos["episode"]["SMP/min"] = smp_rewards.min().item()
                        infos["episode"]["SMP/max"] = smp_rewards.max().item()
                        infos["episode"]["SMP/raw_mean"] = smp_raw_reward.mean().item()
                        infos["episode"]["SMP/raw_min"] = smp_raw_reward.min().item()
                        infos["episode"]["SMP/raw_max"] = smp_raw_reward.max().item()

                    # Accumulate per-env episodic SMP reward step-by-step (like AMP),
                    # logging before reset so completed episodes are captured correctly.
                    all_dones = self.alg.storage.dones[:self.num_steps_per_env].squeeze(-1)  # [T, N]
                    smp_per_step = smp_rewards.squeeze(-1)  # [T, N]
                    for t_idx in range(self.num_steps_per_env):
                        smp_reward_storage += smp_per_step[t_idx]
                        step_dones = all_dones[t_idx]  # [N]
                        # Log before reset (same pattern as AMP)
                        if "episode" in infos:
                            ep_len = infos["episode_length"].to(self.device).float()
                            infos["episode"]["rew_smp"] = smp_reward_storage.mean().item()
                            infos["episode"]["Perstep/rew_smp"] = (smp_reward_storage / ep_len).mean().item()
                        smp_reward_storage[step_dones == 1] = 0.

                    if "episode" in infos:
                        infos["episode"]["total_reward"] += infos["episode"]["rew_smp"]
                        infos["episode"]["Perstep/total_reward"] += infos["episode"]["Perstep/rew_smp"]
                    # Clear buffers for next rollout
                    smp_body_pos_list.clear()
                    smp_body_quat_list.clear()
                    smp_body_vel_list.clear()
                    smp_foot_contact_list.clear()

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs, actions=actions,
                                             infos=infos) # type: ignore

            # call post_rollout method of the environment
            if hasattr(self.env.unwrapped, "post_rollout"):
                self.env.unwrapped.post_rollout()

            # Unfreeze actor after warmup period
            if self.actor_freeze_iters > 0 and (it - start_iter) == self.actor_freeze_iters:
                self.alg.optimizer.param_groups[0]["lr"] = self._actor_lr_backup
                if len(self.alg.optimizer.param_groups) > 2:
                    self.alg.optimizer.param_groups[2]["lr"] = self._other_lr_backup
                print(f"[INFO]: Actor warmup complete at iteration {it}, lr restored to full")

            # Sync normalizers across ranks after rollout collection
            if self.is_distributed and self.empirical_normalization:
                _sync_normalizer(self.obs_normalizer, self.device)
                _sync_normalizer(self.privileged_obs_normalizer, self.device)

            # train policy
            self.alg.policy.train()
            # update policy
            loss_dict = self.alg.update()
            # update policy
            if self.amp_rewards is not None:
                for k in self.amp_rewards.keys():
                    self.amp_rewards[k].train()
                    disc_loss, grad_penalty = self.amp_rewards[k].update()
                    name = '' if k == '' else f"_{k}"
                    loss_dict[f"amp_disc_loss{name}"] = disc_loss
                    loss_dict[f"amp_grad_penalty{name}"] = grad_penalty
                    self.amp_rewards[k].eval()

            # call post_update method of the environment
            if hasattr(self.env.unwrapped, "post_update"):
                env_loss_dict = self.env.unwrapped.post_update()
                if isinstance(env_loss_dict, dict):
                    loss_dict.update(env_loss_dict)

            # schedule
            doing_schedule = hasattr(self.env_unwrapped, "pre_schedule")
            if doing_schedule:
                self.env_unwrapped.pre_schedule()

            if hasattr(self.env_unwrapped, "sync_multi_gpu"):
                self.env_unwrapped.sync_multi_gpu(self.multi_gpu_cfg)

            if doing_schedule:
                schedule = self.env_unwrapped.schedule()
                if hasattr(self.alg.policy, "schedule"):
                    feedback = self.alg.policy.schedule(**schedule)
                else:
                    feedback = {}
                self.env_unwrapped.post_schedule(**feedback)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save mointeraction_contact_datasdel
                if it % self.save_interval == 0:
                    if len(rewbuffer) > 0:
                        current_reward = statistics.mean(rewbuffer)
                        if current_reward > best_reward:
                            best_reward = current_reward
                            self.save(os.path.join(self.log_dir, f"model_best.pt"), remove_extras=False)
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path) # type: ignore

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))


    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                # All-rank reduced ep_info
                if self.is_distributed:
                    import torch.distributed as dist
                    global_value = value.clone()
                    dist.all_reduce(global_value, op=dist.ReduceOp.SUM)
                    global_value = global_value / dist.get_world_size()
                    tag = f"Global/{key}" if "/" in key else f"Global/Episode/{key}"
                    self.writer.add_scalar(tag, global_value, locs["it"])
        
        if hasattr(self.alg.policy, "action_std"):
            mean_std = self.alg.policy.action_std
        else:
            mean_std = 0.0
        if isinstance(mean_std, torch.Tensor):
            mean_std = mean_std.mean().item()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std, locs["it"])
        if hasattr(self.alg.policy, "log_theta") and hasattr(self.alg.policy, "log_sigma"):
            self.writer.add_scalar("Policy/mean_theta", torch.exp(self.alg.policy.log_theta).mean().item(), locs["it"])
            self.writer.add_scalar("Policy/mean_sigma", torch.exp(self.alg.policy.log_sigma).mean().item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            local_mean_reward = statistics.mean(locs["rewbuffer"])
            local_mean_ep_len = statistics.mean(locs["lenbuffer"])
            self.writer.add_scalar("Train/mean_reward", local_mean_reward, locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", local_mean_ep_len, locs["it"])

            # All-rank reduced metrics for distributed training
            if self.is_distributed:
                import torch.distributed as dist
                reward_tensor = torch.tensor([local_mean_reward], device=self.device)
                ep_len_tensor = torch.tensor([local_mean_ep_len], device=self.device)
                count_tensor = torch.tensor([float(len(locs["rewbuffer"]))], device=self.device)
                dist.all_reduce(reward_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(ep_len_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
                global_mean_reward = (reward_tensor / world_size).item()
                global_mean_ep_len = (ep_len_tensor / world_size).item()
                self.writer.add_scalar("Train/global_mean_reward", global_mean_reward, locs["it"])
                self.writer.add_scalar("Train/global_mean_episode_length", global_mean_ep_len, locs["it"])

            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", local_mean_reward, self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", local_mean_ep_len, self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        # MoE: each rank logs its own metrics to its own grouped wandb run.
        # Wandb UI aggregates (mean/min/max) across the group automatically.
        if self.distributed_actor and self.writer is not None:
            local_rew = statistics.mean(locs["rewbuffer"]) if len(locs["rewbuffer"]) > 0 else 0.0
            local_len = statistics.mean(locs["lenbuffer"]) if len(locs["lenbuffer"]) > 0 else 0.0
            self.writer.add_scalar("MoE/reward", local_rew, locs["it"])
            self.writer.add_scalar("MoE/episode_length", local_len, locs["it"])
            if self.expert_metas and "dynamic_score_mean" in self.expert_metas:
                self.writer.add_scalar("MoE/dynamic_score", self.expert_metas["dynamic_score_mean"], locs["it"])

        print(log_string)


    def save(self, path: str, infos=None, remove_extras=True):
        # -- Save model
        # Note: when distributed_critic is enabled, only rank 0 saves (disable_logs is True for other ranks).
        # Rank 0's critic params are saved naturally — no special handling needed.
        saved_dict = {
            "policy_cfg": self.full_policy_cfg,
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if hasattr(self.alg, "critic_optimizer"):
            saved_dict["critic_optimizer_state_dict"] = self.alg.critic_optimizer.state_dict() # type: ignore

        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict() # type: ignore
        # -- Save AMP model if used
        if self.amp_rewards is not None:
            for k in self.amp_rewards.keys():
                name = '' if k == '' else f"_{k}"
                saved_dict[f"amp_state_dict{name}"] = self.amp_rewards[k].network.state_dict()
                saved_dict[f"amp_optimizer_state_dict{name}"] = self.amp_rewards[k].optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # -- Save environment model if used
        if hasattr(self.env.unwrapped, "state_dict"):
            saved_dict["environment_state_dict"] = self.env.unwrapped.state_dict()

        # -- Save expert metadata for distributed_actor training
        if self.expert_metas is not None:
            saved_dict["expert_metas"] = self.expert_metas

        # save model
        torch.save(saved_dict, path)

        # clear extra checkpoints
        if self.max_checkpoint_num is not None and remove_extras:
            files = os.listdir(self.log_dir)
            files = [file for file in files if file.endswith(".pt") and "model" in file]
            files_number = [int(file.split("_")[-1].split(".")[0]) for file in files if 'best' not in file]

            if len(files_number) > self.max_checkpoint_num:
                files_number.sort()
                for file in files_number[:-self.max_checkpoint_num]:
                    path = os.path.join(self.log_dir, f"model_{file}.pt") # type: ignore
                    os.remove(path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs and self.upload_checkpoint:
            self.writer.save_model(path, self.current_learning_iteration) # type: ignore

        # upload to S3 for distributed_actor training
        if self.distributed_actor and self.distributed_s3_prefix:
            self._upload_to_s3(path)

    def _collect_expert_metas(self) -> dict:
        """Collect metadata about this rank's expert (assigned motions, rank info)."""
        metas = {
            "rank": self.gpu_global_rank,
            "world_size": self.gpu_world_size,
        }
        # Get motion names and dynamic scores from the env's motion loader if available
        env_unwrapped = self.env.unwrapped
        if hasattr(env_unwrapped, "motion_loader"):
            ml = env_unwrapped.motion_loader
            if hasattr(ml, "motion_names"):
                metas["motion_names"] = ml.motion_names
                metas["num_motions"] = len(ml.motion_names)
            # Include dynamic score range if sorted by scores
            if hasattr(ml, "distributed_motion_sort") and ml.distributed_motion_sort != "alphabetical":
                try:
                    import json
                    with open(ml.distributed_motion_sort, 'r') as f:
                        scores = json.load(f)
                    rank_scores = [scores[k] for k in ml.motion_names if k in scores]
                    if rank_scores:
                        metas["dynamic_score_min"] = min(rank_scores)
                        metas["dynamic_score_max"] = max(rank_scores)
                        metas["dynamic_score_mean"] = sum(rank_scores) / len(rank_scores)
                except Exception:
                    pass
        return metas

    def _upload_to_s3(self, local_path: str):
        """Upload a checkpoint to S3 in the background."""
        import subprocess
        # Derive run_name from log_dir: log_dir is .../experiment_name/timestamp_runname/rank_N
        # We want the parent of rank_N (or log_dir itself if no rank subdir)
        run_dir = os.path.dirname(self.log_dir) if self.distributed_actor else self.log_dir
        run_name = os.path.basename(run_dir)
        s3_prefix = self.distributed_s3_prefix.replace("{run_name}", run_name)
        s3_path = f"{s3_prefix}/rank_{self.gpu_global_rank}/{os.path.basename(local_path)}"
        cmd = ["aws", "s3", "cp", local_path, s3_path, "--quiet"]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[WARNING]: S3 upload failed for rank {self.gpu_global_rank}: {e}")

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load environment model if used
        if hasattr(self.env.unwrapped, "load_state_dict"):
            self.env.unwrapped.load_state_dict(loaded_dict.get("environment_state_dict", None))
        # -- Load model
        # loaded_dict["model_state_dict"].pop('log_std')
        model_state_dict = loaded_dict["model_state_dict"]
        if self.load_actor_only:
            model_state_dict = {k: v for k, v in model_state_dict.items() if 'actor.' in k}
            
        load_class_name = loaded_dict["policy_cfg"].get('class_name', '')
        if 'Student' in load_class_name and 'Teacher' in load_class_name and self.training_type == "rl":
            model_state_dict = {k.replace('student.', ''): v for k, v in model_state_dict.items() if 'student.' in k}
            self.alg.policy.load_state_dict(model_state_dict, strict=False)
            if self.empirical_normalization:
                try:
                    self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                    print(f"[INFO]: Loaded observation normalizer from: {path}")
                except Exception as e:
                    print(f"[WARNING]: Failed to load observation normalizer. Error: {e}. Reinitializing observation normalizer.")
            print(f"[INFO]: Loaded RL finetuning model from: {path}")
            return loaded_dict["infos"]
        resumed_training = self.alg.policy.load_state_dict(model_state_dict, strict=False)
        if resumed_training and ('Student' not in load_class_name and self.training_type == "distillation"):
            # Distillation using RL model
            resumed_training = False

        # -- Load RND model if used
        if self.alg.rnd:
            try:
                self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
                mismatch_rnd = False
            except Exception as e:
                mismatch_rnd = True
                print(f"[WARNING]: Failed to load RND model. Error: {e}. Initializing new RND model.")
        # -- Load AMP model if used
        if self.amp_rewards is not None:
            amp_loaded = {k: False for k in self.amp_rewards.keys()}
            for k in self.amp_rewards.keys():
                name = '' if k == '' else f"_{k}"
                try:
                    self.amp_rewards[k].network.load_state_dict(loaded_dict[f"amp_state_dict{name}"])
                    amp_loaded[k] = True
                except Exception as e:
                    print(f"[WARNING]: Failed to load AMP model. Error: {e}. Initializing new AMP model.")
            else:
                print("[WARNING]: No AMP model found in the checkpoint. AMP reward will not be loaded.")
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                if not self.load_actor_only:
                    self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # if the training is not resumed but a model is loaded, this run must be distillation training following
                # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                # is not loaded, as the observation space could differ from the previous rl training.
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])

        if not (load_optimizer and resumed_training):
            return loaded_dict["infos"]

        # -- load optimizer if used
        if self.amp_rewards is not None:
            for k in self.amp_rewards.keys():
                if not amp_loaded[k]:
                    continue
                name = '' if k == '' else f"_{k}"
                if f"amp_optimizer_state_dict{name}" in loaded_dict:
                    self.amp_rewards[k].optimizer.load_state_dict(loaded_dict[f"amp_optimizer_state_dict{name}"])
        
        # -- algorithm optimizer
        try:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        except Exception as e:
            print(f"[WARNING]: Failed to load optimizer. Error: {e}. Reinitializing optimizer.")

        # -- RND optimizer if used
        if self.alg.rnd and not mismatch_rnd:
            try:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"]) # type: ignore
            except Exception as e:
                print(f"[WARNING]: Failed to load RND optimizer. Error: {e}. Reinitializing RND optimizer.")

        if hasattr(self.alg, "critic_optimizer") and 'critic_optimizer_state_dict' in loaded_dict:
            # -- critic optimizer
            self.alg.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"]) # type: ignore
                
        # -- load current learning iteration
        # if resumed_training:
        #     self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]
    
    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            def inference_policy(x, *args, **kwargs):
                return self.alg.policy.act_inference(self.obs_normalizer(x), *args, **kwargs)  # noqa: E731
            return inference_policy
        else:
            return policy