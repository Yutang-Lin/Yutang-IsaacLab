# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations: torch.Tensor = None # type: ignore
            self.privileged_observations: torch.Tensor = None # type: ignore
            self.actions: torch.Tensor = None # type: ignore
            self.privileged_actions: torch.Tensor = None # type: ignore
            self.rewards: torch.Tensor = None # type: ignore
            self.dones: torch.Tensor = None # type: ignore
            self.values: torch.Tensor = None # type: ignore
            self.actions_log_prob: torch.Tensor = None # type: ignore
            self.action_mean: torch.Tensor = None # type: ignore
            self.action_sigma: torch.Tensor = None # type: ignore
            self.hidden_states: tuple[torch.Tensor, torch.Tensor] = None # type: ignore
            self.rnd_state: torch.Tensor = None # type: ignore

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
        deterministic=False,
        meta_tensors=None,
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape
        self.deterministic = deterministic

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            if not self.deterministic:
                self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
                self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
                self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        if meta_tensors is not None:
            self.have_meta_tensors = True
            self.create_meta_tensors_storage(meta_tensors)
        else:
            self.have_meta_tensors = False

        # For RND
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(num_transitions_per_env, num_envs, *rnd_state_shape, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0

    def create_meta_tensors_storage(self, meta_tensors: dict[str, torch.Tensor]):
        meta_tensors_list = []
        self._meta_tensors = []

        last_idx = 0
        for k, v in meta_tensors.items():
            assert v.shape[0] == self.num_envs
            num_obs = 1
            for dim in v.shape[1:]:
                num_obs *= dim
            self._meta_tensors.append((k, v.shape[1:], num_obs, last_idx))
            meta_tensors_list.append(torch.zeros(self.num_transitions_per_env, self.num_envs, num_obs, device=self.device))
            last_idx += num_obs
        self.meta_tensors = torch.cat(meta_tensors_list, dim=-1)

    def _assemble_meta_tensors(self, meta_tensors: dict[str, torch.Tensor]):
        for i, (k, v) in enumerate(meta_tensors.items()):
            assert k == self._meta_tensors[i][0], "Meta tensor key mismatch"
            start_idx = self._meta_tensors[i][3]
            end_idx = start_idx + self._meta_tensors[i][2]
            self.meta_tensors[self.step, :, start_idx:end_idx] = v.view(self.num_envs, -1).to(self.device)

    def _disassemble_meta_tensors(self):
        meta_tensors = {}
        for (k, s, n, i) in self._meta_tensors:
            meta_tensors[k] = self.meta_tensors[:, :, i:i+n].view(
                self.num_transitions_per_env, self.num_envs, *s
            )
        return meta_tensors

    def add_transitions(self, transition: Transition, meta_tensors: dict[str, torch.Tensor] = None):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            if not self.deterministic:
                self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
                self.mu[self.step].copy_(transition.action_mean)
                self.sigma[self.step].copy_(transition.action_sigma)

        # For RND
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # For meta observations
        if self.have_meta_tensors:
            self._assemble_meta_tensors(meta_tensors)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i]) # type: ignore

    def clear(self):
        self.step = 0

    def compute_td_returns(self, last_values, gamma):
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            self.returns[step] = self.rewards[step] + next_is_not_terminal * gamma * next_values

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # for distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield self.observations[i], privileged_observations, self.actions[i], self.privileged_actions[
                i
            ], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        advantages = self.advantages.flatten(0, 1)
        if not self.deterministic:
            old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
            old_mu = self.mu.flatten(0, 1)
            old_sigma = self.sigma.flatten(0, 1)

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state.flatten(0, 1)

        # For meta observations
        if self.have_meta_tensors:
            meta_tensors = self._disassemble_meta_tensors()
            for k, v in meta_tensors.items():
                meta_tensors[k] = v.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                privileged_observations_batch = privileged_observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                advantages_batch = advantages[batch_idx]
                if not self.deterministic:
                    old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                    old_mu_batch = old_mu[batch_idx]
                    old_sigma_batch = old_sigma[batch_idx]
                else:
                    old_actions_log_prob_batch = None
                    old_mu_batch = None
                    old_sigma_batch = None

                # -- For meta observations
                if self.have_meta_tensors:
                    meta_tensors_batch = {}
                    for k, v in meta_tensors.items():
                        meta_tensors_batch[k] = v[batch_idx]
                else:
                    meta_tensors_batch = None

                # -- For RND
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]
                else:
                    rnd_state_batch = None

                # yield the mini-batch
                yield obs_batch, privileged_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None, rnd_state_batch, meta_tensors_batch

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_privileged_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_privileged_obs_trajectories = padded_obs_trajectories

        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        # For meta observations
        if self.have_meta_tensors:
            padded_meta_tensors_trajectories = self._disassemble_meta_tensors()
            # for k, v in padded_meta_tensors_trajectories.items():
            #     padded_meta_tensors_trajectories[k], _ = split_and_pad_trajectories(v, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None

                actions_batch = self.actions[:, start:stop]
                if not self.deterministic:
                    old_mu_batch = self.mu[:, start:stop]
                    old_sigma_batch = self.sigma[:, start:stop]
                else:
                    old_mu_batch = None
                    old_sigma_batch = None
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                if not self.deterministic:
                    old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]
                else:
                    old_actions_log_prob_batch = None

                # For meta observations
                if self.have_meta_tensors:
                    meta_tensors_batch = {}
                    for k, v in padded_meta_tensors_trajectories.items():
                        meta_tensors_batch[k] = v[:, start:stop]
                else:
                    meta_tensors_batch = None

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a # type: ignore
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c # type: ignore
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, privileged_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch, rnd_state_batch, meta_tensors_batch

                first_traj = last_traj
