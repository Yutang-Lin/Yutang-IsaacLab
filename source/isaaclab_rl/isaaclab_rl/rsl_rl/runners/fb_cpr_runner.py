# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from tqdm import tqdm
import statistics
import time
import torch
import numpy as np

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import store_code_state

from ..fb_cpr import (
    Actor,
    DoubleQCritic,
    FBNetwork,
    Discriminator
)
from ..storage.fb_cpr_storage import FBCPRStorage
from ..utils import broadcast_parameters, reduce_gradients

from collections import deque
from copy import deepcopy

class FBCPRRunner(OnPolicyRunner):
    """FBCPR runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg

        self.device = device
        self.env = env
        self.env_unwrapped = env.unwrapped # type: ignore

        # set manual termination
        self.env_unwrapped.manual_termination = True

        # loss alphas
        self.alpha_disc = self.cfg.get('alpha_disc', 0.05)
        self.alpha_auxi = self.cfg.get('alpha_auxi', 0.02)
        self.gp_loss_weight = self.cfg.get('gp_weight', 10.0)
        self.ortho_loss_weight = self.cfg.get('ortho_loss_weight', 0.01)

        # relabel configs
        self.p_relable = self.cfg.get('p_relable', 0.6)
        self.p_goal_reaching = self.cfg.get('p_goal_reaching', 0.2)
        self.p_traj_following = self.cfg.get('p_traj_following', 0.2)
        self.p_random = self.cfg.get('p_random', 0.2)

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # check if checkpoint should be uploaded
        self.upload_checkpoint = self.cfg.get("upload_checkpoint", True)

        # max checkpoint number
        self.max_checkpoint_num = self.cfg.get("max_checkpoint_num", 10)

        # action clip range
        self.action_clip_range = self.cfg.get("action_clip_range", [-50.0, 50.0])

        # resolve dimensions of observations
        obs_dict = self.env_unwrapped._get_observations(compute_meta=True)
        meta_tensors = self.env_unwrapped._get_meta_tensors()
        if len(meta_tensors) == 0:
            meta_tensors = None
        else:
            print(f"[INFO]: Meta tensors are used, keys: {meta_tensors.keys()}")
        num_obs = obs_dict['policy'].shape[1]

        # resolve meta dict
        meta_dict = dict(
            actor_obs_meta=obs_dict['policy_meta'],
            critic_obs_meta=obs_dict['critic_meta'],
        )

        # resolve dimensions of privileged observations
        num_state = obs_dict['critic'].shape[1]

        self.full_policy_cfg = deepcopy(self.policy_cfg)
        self.full_policy_cfg["_args"] = [num_obs, num_state, self.env.num_actions]
        self.full_policy_cfg.update(meta_dict)

        # define networks
        self.dim_latent = 256
        self.actor = Actor(num_obs, self.env.num_actions, self.dim_latent)
        self.fb_network = FBNetwork(num_state, self.env.num_actions, self.dim_latent,
                                    ortho_loss_weight=self.ortho_loss_weight)
        self.q_disc = DoubleQCritic(num_state, self.env.num_actions, self.dim_latent)
        self.q_auxi = DoubleQCritic(num_state, self.env.num_actions, self.dim_latent)
        self.discriminator = Discriminator(num_state, self.env.num_actions, self.dim_latent,
                                           grad_penalty_weight=self.gp_loss_weight)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam([
            {"params": self.fb_network.parameters(), "lr": 1e-4},
            {"params": self.discriminator.parameters(), "lr": 1e-4},
            {"params": self.q_disc.parameters(), "lr": 1e-4},
            {"params": self.q_auxi.parameters(), "lr": 1e-4},
        ])

        # storage
        self.episode_length = self.cfg.get("episode_length", 500)
        self.storage = FBCPRStorage(num_envs=self.env.num_envs,
                                    num_buffer=self.cfg.get("num_buffer", 100),
                                    episode_length=self.episode_length,
                                    dim_obs=num_obs,
                                    dim_state=num_state,
                                    dim_action=self.env.num_actions,
                                    dim_latent=self.dim_latent,
                                    device=self.device)
        
        # transition
        self.transition = self.storage.Transition()

        # training
        self.n_ups = self.cfg.get("n_ups", 16)
        self.t_seq = self.cfg.get("t_seq", 8)
        self.batch_size = self.cfg.get("batch_size", 1024)

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
    
    @torch.no_grad()
    def _sample_expert_states(self,
                              batch_size: int):
        """
        Sample the expert states.
        """
        pass

    @torch.no_grad()
    def _sample_latent(self, batch_size: int):
        """
        Sample the latent.
        """
        latent = torch.randn(batch_size, self.dim_latent, device=self.device)
        env_type = torch.randint(0, 
                    self.env_unwrapped.num_env_types, (batch_size,), device=self.device)
        env_time = torch.rand(batch_size, device=self.device)
        # Sample a random number
        random_number = torch.rand(batch_size, device=latent.device)
        
        # Goal reaching
        goal_reaching = (random_number < self.p_goal_reaching)
        goal_reaching_ids = goal_reaching.nonzero().squeeze(-1)
        if len(goal_reaching_ids) > 0:
            goal_state, goal_env_type, goal_env_time = \
                self.storage.sample_batched_state(len(goal_reaching_ids))
            goal_latent = self.fb_network.encode_trajectory(goal_state)
            latent[goal_reaching_ids] = goal_latent
            env_type[goal_reaching_ids] = goal_env_type
            env_time[goal_reaching_ids] = goal_env_time
        
        # Traj following
        traj_following = (random_number < self.p_traj_following + self.p_goal_reaching) & (~goal_reaching)
        traj_following_ids = traj_following.nonzero().squeeze(-1)
        if len(traj_following_ids) > 0:
            expert_states, expert_env_type, expert_env_time = \
                self._sample_expert_states(len(traj_following_ids))
            expert_latent = self.fb_network.encode_trajectory(expert_states)
            latent[traj_following_ids] = expert_latent
            env_type[traj_following_ids] = expert_env_type
            env_time[traj_following_ids] = expert_env_time
        
        latent = (np.sqrt(self.dim_latent) * latent) / latent.norm(dim=-1, keepdim=True)
        return latent, env_type, env_time

    @torch.inference_mode()
    def _rollout_full_trajectory(self):
        """
        Rollout the full trajectory.
        """
        # Sample latent
        latent, env_type, env_time = self._sample_latent(self.env.num_envs)
        # Clear the storage
        self.storage.clear_episode()
        # Reset the environment
        obs, infos = self.env.reset(env_type=env_type, env_time=env_time) # type: ignore
        state = infos["observations"]["critic"]
        # Accumulative reward
        cum_reward = torch.zeros(self.env.num_envs, device=self.device)
        # Rollout the trajectory
        for t in range(self.episode_length):
            # Sample action
            action = self.actor(obs, latent)
            # Step the environment
            next_obs, reward, dones, info = self.env.step(action)
            # Update the state
            next_state = info["observations"]["critic"]
            # Update the accumulative reward
            cum_reward += reward
            # Add the transition to the storage
            self.transition.observations = obs
            self.transition.next_observations = next_obs
            self.transition.states = state
            self.transition.next_states = next_state
            self.transition.actions = action
            self.transition.rewards = reward
            self.transition.latents = latent
            self.storage.add_transitions(self.transition)
            # Clear the transition
            self.transition.clear()
            # Update the observation and state
            obs = next_obs
            state = next_state
            # Yeild for control
            yield t, cum_reward
        self.storage.finish_episode(env_type, env_time)
    
    @torch.no_grad()
    def _relabel_latent(self,
                        latent: torch.Tensor,
                        batched_state: torch.Tensor,
                        batched_expert_latent: torch.Tensor) -> torch.Tensor:
        """
        Relabel the latent with mixture of latents.
        """
        new_latent = latent.clone()
        # Sample a random number
        random_number = torch.rand(latent.shape[0], device=latent.device)
        
        # Goal reaching
        goal_reaching = (random_number < self.p_goal_reaching)
        goal_reaching_ids = goal_reaching.nonzero().squeeze(-1)
        if len(goal_reaching_ids) > 0:
            goal = torch.randint(0, batched_state.shape[0], 
                                 (len(goal_reaching_ids),), device=batched_state.device)
            goal = self.fb_network.encode_trajectory(batched_state[goal])
            new_latent[goal_reaching_ids] = goal
        
        # Traj following
        traj_following = (random_number < self.p_traj_following + self.p_goal_reaching) & (~goal_reaching)
        traj_following_ids = traj_following.nonzero().squeeze(-1)
        if len(traj_following_ids) > 0:
            traj = torch.randint(0, batched_expert_latent.shape[0], 
                                 (len(traj_following_ids),), device=batched_expert_latent.device)
            new_latent[traj_following_ids] = batched_expert_latent[traj]
        
        # Random relabel
        random_relabel = (random_number < self.p_random + self.p_traj_following + self.p_goal_reaching) & (~goal_reaching) & (~traj_following)
        random_relabel_ids = random_relabel.nonzero().squeeze(-1)
        if len(random_relabel_ids) > 0:
            new_latent[random_relabel_ids] = torch.randn_like(latent[random_relabel_ids])
        
        # ensure the latent is normalized to hyper ball
        new_latent = (np.sqrt(self.dim_latent) * new_latent) / new_latent.norm(dim=-1, keepdim=True)
        return new_latent
    
    @torch.no_grad()
    def _ER_FB(self,
               batched_expert_states: torch.Tensor) -> torch.Tensor:
        """
        Compute the ER-FB loss.
        """
        return self.fb_network.encode_trajectory(batched_expert_states)

    def _update_critic_with_batch(self, 
                                batched_state: torch.Tensor,
                                batched_next_state: torch.Tensor,
                                batched_next_obs: torch.Tensor,
                                batched_action: torch.Tensor,
                                batched_reward: torch.Tensor,
                                batched_latent: torch.Tensor,
                                batched_expert_states: torch.Tensor):
        """
        Update the networks with a batch of data.
        """
        # Compute the expert latent
        z_j = self._ER_FB(batched_expert_states)
        # Compute the discriminator loss
        discrim_loss = self.discriminator.loss(batched_state, 
                                               batched_latent, 
                                               batched_expert_states.flatten(0, 1),
                                               z_j.repeat_interleave(batched_expert_states.shape[1], dim=0)
                                            )
        # Relabel the latent
        z_i = self._relabel_latent(batched_latent, batched_state, z_j)
        # Compute a_prime
        with torch.no_grad():
            a_prime_no_grad = self.actor(batched_next_obs, z_i)
        # Compute the FB network loss
        fb_loss = self.fb_network.loss(
            batched_state, 
            batched_next_state, 
            batched_action, 
            a_prime_no_grad, 
            z_i)
        # Compute the discriminator reward
        discrim_reward = self.discriminator.compute_reward(batched_state, z_i)
        # Compute the Q critic loss
        q_disc_loss = self.q_disc.loss(batched_state, 
                                       batched_action, 
                                       discrim_reward, 
                                       z_i, 
                                       batched_next_state, 
                                       a_prime_no_grad)
        q_auxi_loss = self.q_auxi.loss(batched_state, 
                                       batched_action, 
                                       batched_reward, 
                                       z_i, 
                                       batched_next_state, 
                                       a_prime_no_grad)
        # Compute the total loss
        loss_dict = dict(
            discrim_loss=discrim_loss,
            q_disc_loss=q_disc_loss,
            q_auxi_loss=q_auxi_loss,
            **fb_loss,
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = torch.tensor(0.0, device=self.device)
        for key, value in loss_dict.items():
            critic_loss += value
        critic_loss.backward()
        # Reduce gradients
        if self.gpu_world_size > 1:
            reduce_gradients(self.fb_network)
            reduce_gradients(self.discriminator)
            reduce_gradients(self.q_disc)
            reduce_gradients(self.q_auxi)
        # Step the optimizer
        self.critic_optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}
    
    def _update_actor_with_batch(self,
                                 batched_obs: torch.Tensor,
                                 batched_latent: torch.Tensor):
        """
        Update the actor with a batch of data.
        """
        # Compute the action
        action = self.actor(batched_obs, batched_latent)
        # Compute the Q values
        q_fb = self.fb_network.as_q_function(batched_obs, action, batched_latent).mean()
        q_disc = self.q_disc.as_q_function(batched_obs, action, batched_latent).mean()
        q_auxi = self.q_auxi.as_q_function(batched_obs, action, batched_latent).mean()
        q_sum = q_fb + q_disc + q_auxi
        # Compute the loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss = -q_sum.mean()
        actor_loss.backward()
        # Reduce gradients
        if self.gpu_world_size > 1:
            reduce_gradients(self.actor)
        # Step the optimizer
        self.actor_optimizer.step()
        return {
            'actor_q_fb_loss': q_fb.item(),
            'actor_q_disc_loss': q_disc.item(),
            'actor_q_auxi_loss': q_auxi.item(),
            'actor_loss': q_sum.item(),
        }


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

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter # type: ignore

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # Broadcast parameters
        if self.gpu_world_size > 1:
            broadcast_parameters(self.fb_network)
            broadcast_parameters(self.discriminator)
            broadcast_parameters(self.q_disc)
            broadcast_parameters(self.q_auxi)
            broadcast_parameters(self.actor)

        best_iteration = [-float("inf"), -1]
        # prefilling the buffer
        for _ in range(1):
            for t, cum_reward in tqdm(self._rollout_full_trajectory(), 
                                        total=self.env.num_envs, 
                                        desc="Prefilling the buffer"):
                pass
            best_iteration[0] = max(best_iteration[0], cum_reward.mean().item())

        # start learning
        rollout_collector = self._rollout_full_trajectory()
        for it in range(1, num_learning_iterations + 1):
            self.current_learning_iteration = it
            # step environment
            start_time = time.time()
            try:
                t, cum_reward = next(rollout_collector)
            except StopIteration:
                best_iteration[0] = max(best_iteration[0], cum_reward.mean().item())
                rollout_collector = self._rollout_full_trajectory()
            collection_time = time.time() - start_time
            # update with gradient
            loss_dict = dict()
            for _ in range(self.n_ups):
                # sample batch
                (
                 batch_obs,
                 batch_next_obs, 
                 batch_state, 
                 batch_next_state, 
                 batch_action, 
                 batch_reward, 
                 batch_latent
                ) = self.storage.sample_batch(self.batch_size)
                # sample expert states
                batch_expert_states, _, _ = self._sample_expert_states(self.batch_size)
                # update critic
                critic_loss_dict = self._update_critic_with_batch(
                                                batch_state,
                                                batch_next_state,
                                                batch_next_obs,
                                                batch_action,
                                                batch_reward,
                                                batch_latent,
                                                batch_expert_states
                                            )
                # update actor
                actor_loss_dict = self._update_actor_with_batch(batch_obs, batch_latent)
                actor_loss_dict.update(critic_loss_dict)
                for key, value in actor_loss_dict.items():
                    if not key in loss_dict:
                        loss_dict[key] = 0.0
                    loss_dict[key] += value
            loss_dict = {k: v / self.n_ups for k, v in loss_dict.items()}
            learn_time = time.time() - start_time - collection_time
            
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if best_iteration[1] == it:
                    self.save(os.path.join(self.log_dir, f"model_best.pt"), remove_extras=False)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        assert self.writer is not None, "Writer is not initialized"
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        # if locs["ep_infos"]:
        #     for key in locs["ep_infos"][0]:
        #         infotensor = torch.tensor([], device=self.device)
        #         for ep_info in locs["ep_infos"]:
        #             # handle scalar and zero dimensional tensor infos
        #             if key not in ep_info:
        #                 continue
        #             if not isinstance(ep_info[key], torch.Tensor):
        #                 ep_info[key] = torch.Tensor([ep_info[key]])
        #             if len(ep_info[key].shape) == 0:
        #                 ep_info[key] = ep_info[key].unsqueeze(0)
        #             infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        #         value = torch.mean(infotensor)
        #         # log to logger and terminal
        #         if "/" in key:
        #             self.writer.add_scalar(key, value, locs["it"])
        #             ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
        #         else:
        #             self.writer.add_scalar("Episode/" + key, value, locs["it"])
        #             ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
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
        # self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std, locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        # if len(locs["rewbuffer"]) > 0:
        #     # everything else
        #     self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
        #     self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
        #     if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
        #         self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
        #         self.writer.add_scalar(
        #             "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
        #         )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        # if len(locs["rewbuffer"]) > 0:
        #     log_string = (
        #         f"""{'#' * width}\n"""
        #         f"""{str.center(width, ' ')}\n\n"""
        #         f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
        #             'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
        #         f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
        #     )
        #     # -- Losses
        #     for key, value in locs["loss_dict"].items():
        #         log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
        #     # -- Rewards
        #     if self.alg.rnd:
        #         log_string += (
        #             f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
        #             f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
        #         )
        #     log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
        #     # -- episode info
        #     log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        # else:
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
        print(log_string)

    
    def save(self, path: str, infos=None, remove_extras=True):
        # -- Save model
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

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        # loaded_dict["model_state_dict"].pop('log_std')
        model_state_dict = loaded_dict["model_state_dict"]
        load_class_name = loaded_dict["policy_cfg"].get('class_name', '')
        if 'Student' in load_class_name and 'Teacher' in load_class_name and self.training_type == "rl":
            model_state_dict = {k.replace('student.', ''): v for k, v in model_state_dict.items() if 'student.' in k}
            self.alg.policy.load_state_dict(model_state_dict, strict=False)
            print(f"[INFO]: Loaded RL finetuning model from: {path}")
            return loaded_dict["infos"]
        resumed_training = self.alg.policy.load_state_dict(model_state_dict, strict=False)

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
        self.actor.train()
        self.fb_network.train()
        self.discriminator.train()
        self.q_disc.train()
        self.q_auxi.train()

    def eval_mode(self):
        self.actor.eval()
        self.fb_network.eval()
        self.discriminator.eval()
        self.q_disc.eval()
        self.q_auxi.eval()

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