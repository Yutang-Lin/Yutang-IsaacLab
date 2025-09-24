# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
# rsl-rl
from isaaclab_rl.rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from isaaclab_rl.rsl_rl.storage import FlowDAggerStorage


class FlowDAgger:
    """Flow DAgger algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        device="cpu",
        flow_state_horizon=None,
        flow_state_normalizer=None,
        allow_amp=False,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.device_type = torch.device(self.device).type
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.allow_amp = allow_amp
        self.amp_dtype = torch.bfloat16 if allow_amp else torch.float32
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class
        self.flow_state_horizon = flow_state_horizon
        self.flow_state_normalizer = flow_state_normalizer

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, fused=True)
        # if self.allow_amp:
        #     self.grad_scaler = amp.GradScaler(device=self.device)
        # else:
        #     self.grad_scaler = None
        self.grad_scaler = None # replaced by fused adam
        self.transition = FlowDAggerStorage.Transition()
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # flow state parameters
        self.flow_state_storage = None

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape, 
        meta_tensors=None
    ):
        # create rollout storage
        self.storage = FlowDAggerStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
            meta_tensors=meta_tensors,
            flow_state_horizon=self.flow_state_horizon,
        )

    def act(self, obs, teacher_obs, infos=None, **kwargs):
        # compute the actions
        self.transition.actions = self.policy.act(obs).detach() # type: ignore
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        if infos is not None and 'robot_state' in infos:
            self.transition.flow_state = infos['robot_state']
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        assert 'robot_state' in infos, "Robot state must be provided for Flow DAgger."
        if self.transition.flow_state is None:
            self.transition.flow_state = infos['robot_state']
        # record the transition
        self.storage.add_transitions(self.transition, 
                                     meta_tensors=infos.get('meta_tensors', None))
        self.transition.clear()
        self.policy.reset(dones)

    def _inner_update(self):
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_extra_loss = {}
        loss = 0
        cnt = 0

        if hasattr(self.policy, "pre_train"):
            self.policy.pre_train()

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, student_actions, privileged_actions, dones, (flow_state, flow_dones) in self.storage.generator():

                # inference the student for gradient computation
                obs = obs.to(self.amp_dtype)
                student_actions = student_actions.to(self.amp_dtype)
                privileged_actions = privileged_actions.to(self.amp_dtype)
                if flow_state is not None:
                    flow_state = flow_state.to(self.amp_dtype)

                # forward the policy
                actions = self.policy.act_inference(obs)

                # normalize the flow state
                with torch.inference_mode():
                    unwrapped_env = getattr(self, "unwrapped_env", None)
                    if flow_state is not None and unwrapped_env is not None:
                        normalizer = getattr(unwrapped_env, self.flow_state_normalizer, None)
                        assert normalizer is not None, "Flow state normalizer must be provided for Flow DAgger."
                        flow_state = normalizer(flow_state)

                # compute the extra loss
                extra_loss = self.policy.extra_loss(
                    student_actions_batch=student_actions,
                    teacher_actions_batch=privileged_actions,
                    flow_state_batch=flow_state,
                    flow_dones_batch=flow_dones,
                )
                if isinstance(extra_loss, tuple):
                    extra_loss, value_dict = extra_loss
                else:
                    value_dict = {}
                for key, value in extra_loss.items():
                    loss = loss + value
                    if key not in mean_extra_loss:
                        mean_extra_loss[key] = 0
                    if key in value_dict:
                        mean_extra_loss[key] += value_dict[key]
                    else:
                        mean_extra_loss[key] += value.item()

                # behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    if self.grad_scaler is not None:
                        self.grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    if self.grad_scaler is not None:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        if hasattr(self.policy, "after_train"):
            self.policy.after_train()

        mean_behavior_loss /= cnt
        for key, value in mean_extra_loss.items():
            mean_extra_loss[key] /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss,
                     **mean_extra_loss}

        return loss_dict

    def update(self):
        with amp.autocast(device_type=self.device_type, dtype=self.amp_dtype):
            loss_dict = self._inner_update()
        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
