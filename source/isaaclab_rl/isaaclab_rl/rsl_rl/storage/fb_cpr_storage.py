# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

class FBCPRStorage:
    class Transition:
        def __init__(self):
            self.observations: torch.Tensor = None # type: ignore
            self.next_observations: torch.Tensor = None # type: ignore
            self.states: torch.Tensor = None # type: ignore
            self.next_states: torch.Tensor = None # type: ignore
            self.actions: torch.Tensor = None # type: ignore
            self.rewards: torch.Tensor = None # type: ignore
            self.latents: torch.Tensor = None # type: ignore

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_buffer,
        episode_length,
        dim_obs,
        dim_state,
        dim_action,
        dim_latent,
        device="cpu",
        meta_tensors=None,
    ):
        # store inputs
        self.device = device
        self.num_buffer = num_buffer
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.dim_obs = dim_obs
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_latent = dim_latent

        # Core
        self.observations = torch.zeros(num_buffer, num_envs, episode_length, dim_obs, device=self.device)
        self.next_observations = torch.zeros(num_buffer, num_envs, episode_length, dim_obs, device=self.device)
        self.states = torch.zeros(num_buffer, num_envs, episode_length, dim_state, device=self.device)
        self.next_states = torch.zeros(num_buffer, num_envs, episode_length, dim_state, device=self.device)
        self.actions = torch.zeros(num_buffer, num_envs, episode_length, dim_action, device=self.device)
        self.latents = torch.zeros(num_buffer, num_envs, episode_length, dim_latent, device=self.device)
        self.rewards = torch.zeros(num_buffer, num_envs, episode_length, 1, device=self.device)
        self.env_type = torch.zeros(num_buffer, num_envs, 1, dtype=torch.long, device=self.device)
        self.env_time = torch.zeros(num_buffer, num_envs, 1, device=self.device)

        # if meta_tensors is not None:
        #     self.have_meta_tensors = True
        #     self.create_meta_tensors_storage(meta_tensors)
        # else:
        #     self.have_meta_tensors = False

        # counter for the number of transitions stored
        self.buffer_step = 0
        self.episode_step = 0

        # counter for indicating valid datas
        self.valid_buffer_count = 0

    # def create_meta_tensors_storage(self, meta_tensors: dict[str, torch.Tensor]):
    #     meta_tensors_list = []
    #     self._meta_tensors = []

    #     last_idx = 0
    #     for k, v in meta_tensors.items():
    #         assert v.shape[0] == self.num_envs
    #         num_obs = 1
    #         for dim in v.shape[1:]:
    #             num_obs *= dim
    #         self._meta_tensors.append((k, v.shape[1:], num_obs, last_idx))
    #         meta_tensors_list.append(torch.zeros(self.num_buffer, self.num_envs, num_obs, device=self.device))
    #         last_idx += num_obs
    #     self.meta_tensors = torch.cat(meta_tensors_list, dim=-1)

    # def _assemble_meta_tensors(self, meta_tensors: dict[str, torch.Tensor]):
    #     for i, (k, v) in enumerate(meta_tensors.items()):
    #         assert k == self._meta_tensors[i][0], "Meta tensor key mismatch"
    #         start_idx = self._meta_tensors[i][3]
    #         end_idx = start_idx + self._meta_tensors[i][2]
    #         self.meta_tensors[self.step, :, start_idx:end_idx] = v.view(self.num_envs, -1).to(self.device)

    # def _disassemble_meta_tensors(self):
    #     meta_tensors = {}
    #     for (k, s, n, i) in self._meta_tensors:
    #         meta_tensors[k] = self.meta_tensors[:, :, i:i+n].view(
    #             self.num_buffer, self.num_envs, *s
    #         )
    #     return meta_tensors

    def add_transitions(self, transition: Transition, meta_tensors: dict[str, torch.Tensor] = None): # type: ignore
        # check if the transition is valid
        if self.episode_step >= self.episode_length:
            raise OverflowError("Episode buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.buffer_step, :, self.episode_step].copy_(transition.observations)
        self.next_observations[self.buffer_step, :, self.episode_step].copy_(transition.next_observations)
        self.states[self.buffer_step, :, self.episode_step].copy_(transition.states)
        self.next_states[self.buffer_step, :, self.episode_step].copy_(transition.next_states)
        self.actions[self.buffer_step, :, self.episode_step].copy_(transition.actions)
        self.rewards[self.buffer_step, :, self.episode_step].copy_(transition.rewards.view(self.num_envs, 1))
        self.latents[self.buffer_step, :, self.episode_step].copy_(transition.latents)

        # For meta observations
        # if self.have_meta_tensors:
        #     self._assemble_meta_tensors(meta_tensors)

        # increment the counter
        self.episode_step += 1

    def clear_episode(self):
        self.episode_step = 0

    def finish_episode(self, env_type: torch.Tensor, env_time: torch.Tensor):
        # update environment id and time
        self.env_type[self.buffer_step, :, 0] = env_type
        self.env_time[self.buffer_step, :, 0] = env_time

        # reset episode step and buffer step
        self.episode_step = 0
        self.buffer_step += 1
        if self.buffer_step >= self.num_buffer:
            self.buffer_step = 0
        self.valid_buffer_count += 1

    def sample_batch(self, batch_size: int):
        assert self.valid_buffer_count > 0, "No valid data in the buffer, fill it first!"
        indices_buffer = torch.randint(0, self.valid_buffer_count, (batch_size,))
        indices_env = torch.randint(0, self.num_envs, (batch_size,))
        indices_episode = torch.randint(0, self.episode_length, (batch_size,))
        return (
            self.observations[indices_buffer, indices_env, indices_episode], 
            self.next_observations[indices_buffer, indices_env, indices_episode], 
            self.states[indices_buffer, indices_env, indices_episode], 
            self.next_states[indices_buffer, indices_env, indices_episode], 
            self.actions[indices_buffer, indices_env, indices_episode], 
            self.rewards[indices_buffer, indices_env, indices_episode], 
            self.latents[indices_buffer, indices_env, indices_episode],
        )
    
    def sample_batched_state(self, batch_size: int):
        assert self.valid_buffer_count > 0, "No valid data in the buffer, fill it first!"
        indices_buffer = torch.randint(0, self.valid_buffer_count, (batch_size,))
        indices_env = torch.randint(0, self.num_envs, (batch_size,))
        indices_episode = torch.randint(0, self.episode_length, (batch_size,))

        return (
            self.states[indices_buffer, indices_env, indices_episode],
            self.env_type[indices_buffer, indices_env, 0],
            self.env_time[indices_buffer, indices_env, 0],
        )
    
if __name__ == "__main__":
    storage = FBCPRStorage(num_envs=1024, 
                           num_buffer=10, 
                           episode_length=500, 
                           dim_obs=256, 
                           dim_state=512, 
                           dim_action=29, 
                           dim_latent=256)
    import time
    start_time = time.time()
    for i in range(10):
        batch = storage.sample_batch(1024)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")