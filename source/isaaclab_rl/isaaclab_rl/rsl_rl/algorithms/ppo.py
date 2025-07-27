import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms.ppo import PPO as RslRlPPO
from isaaclab_rl.rsl_rl.modules import ActorCritic

from itertools import chain
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.utils import string_to_callable
from isaaclab_rl.rsl_rl.storage import RolloutStorage

class PPO(RslRlPPO):
    policy: ActorCritic

    def __init__(self, policy,
        num_learning_epochs=1,
        num_critic_extra_epochs=0,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        *args, **kwargs):

        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract learning rate and remove it from the original dict
            learning_rate = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=learning_rate)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        policy_named_parameters = list(self.policy.named_parameters())
        actor_params = [p for n, p in policy_named_parameters if "actor" in n]
        critic_params = [p for n, p in policy_named_parameters if "critic" in n]
        other_params = [p for n, p in policy_named_parameters if "actor" not in n and "critic" not in n]
        self.optimizer = optim.Adam([
            {"params": actor_params, "lr": learning_rate},
            {"params": critic_params, "lr": learning_rate},
            {"params": other_params, "lr": learning_rate}
        ], lr=learning_rate, eps=1e-5)

        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_critic_extra_epochs = num_critic_extra_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.desired_clipping = kwargs.pop("desired_clipping", 0.0)
        self.importance_sample_value = kwargs.pop("importance_sample_value", False)
        self.centralize_log_prob = kwargs.pop("centralize_log_prob", False)
        self.use_lipschitz_constraint = kwargs.pop("use_lipschitz_constraint", False)
        self.lipschitz_constraint_coef = kwargs.pop("lipschitz_constraint_coef", 2e-2)
        self.adjust_critic_lr = kwargs.pop("adjust_critic_lr", True)

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach() # type: ignore
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(
            self.transition.actions,
            collecting=True
        ).detach()

        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def compute_returns(self, last_critic_obs, **kwargs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def _compute_lipschitz_constraint(self, obs_batch: torch.Tensor, action_mean_batch: torch.Tensor):
        grad_log_prob = torch.autograd.grad(action_mean_batch.sum(), obs_batch, create_graph=True,
                                            allow_unused=True)[0]
        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
        return gradient_penalty_loss

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_extra_loss = {}
        mean_kl = 0

        # -- Lipschitz constraint loss
        if self.use_lipschitz_constraint:
            mean_extra_loss["lipschitz_constraint"] = 0

        # to track the ratio of clipped and unclipped ratios
        num_all_ratios = 0
        num_clipped_ratios = 0
        num_updates = 0

        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs + self.num_critic_extra_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs + self.num_critic_extra_epochs)

        if hasattr(self.policy, "pre_train"):
            self.policy.pre_train()

        # iterate over batches
        for update_id, (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in enumerate(generator):
            if update_id // self.num_mini_batches > self.num_learning_epochs:
                # critic extra epochs
                value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = self.value_loss_coef * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                continue

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            if self.use_lipschitz_constraint:
                actor_obs_batch = obs_batch.clone()
                actor_obs_batch.requires_grad_()
            else:
                actor_obs_batch = obs_batch

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(actor_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Surrogate loss
            if self.centralize_log_prob:
                log_prob_shift = old_actions_log_prob_batch.mean() - actions_log_prob_batch.mean()
                log_ratio = actions_log_prob_batch + log_prob_shift - torch.squeeze(old_actions_log_prob_batch) # type: ignore
                ratio = torch.exp(log_ratio)
            else:
                log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch) # type: ignore
                ratio = torch.exp(log_ratio)
            
            # KL
            # NOTE: using stablebaseline3 implementation
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    # NOTE: using stablebaseline3 implementation
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) # type: ignore
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    ) # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    kl_mean = kl_mean.item()
                    mean_kl += kl_mean
                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    # NOTE: using stablebaseline3 implementation
                    if self.desired_clipping < 1e-3:
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        # Update the learning rate for all GPUs
                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()

                        # Update the learning rate for all parameter groups
                        # NOTE: 0 is actor, 1 is critic, 2 is other
                        self.optimizer.param_groups[0]["lr"] = self.learning_rate
                        if self.adjust_critic_lr:
                            self.optimizer.param_groups[1]["lr"] = self.learning_rate
                        self.optimizer.param_groups[2]["lr"] = self.learning_rate

                    # NOTE: using stablebaseline3 implementation
                    # if kl_mean > self.desired_kl * 5.0:
                    #     self.learning_rate = max(1e-5, self.learning_rate / 4.0)
                    #     break # stop training if KL-divergence is too high

            if hasattr(self.policy, "extra_loss"):
                extra_loss = self.policy.extra_loss(
                    obs_batch=obs_batch,
                    critic_obs_batch=critic_obs_batch,
                )
                for key, value in extra_loss.items():
                    if key not in mean_extra_loss:
                        mean_extra_loss[key] = 0.0
                    mean_extra_loss[key] += value.item()

            num_all_ratios += ratio.numel()
            num_clipped_ratios += torch.abs(ratio - 1.0).gt(self.clip_param).sum().item()
            
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.importance_sample_value:
                returns_batch = returns_batch * ratio.detach().clamp(
                    1 - self.clip_param, 1 + self.clip_param
                )
                
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if self.use_lipschitz_constraint:
                action_mean_batch = self.policy.action_mean
                lipschitz_constraint_loss = self._compute_lipschitz_constraint(actor_obs_batch,
                                                                               action_mean_batch)
                mean_extra_loss["lipschitz_constraint"] += lipschitz_constraint_loss.item() * \
                                                             self.lipschitz_constraint_coef
            else:
                lipschitz_constraint_loss = 0.0

            loss = surrogate_loss + \
                self.value_loss_coef * value_loss - \
                self.entropy_coef * entropy_batch.mean() + \
                self.lipschitz_constraint_coef * lipschitz_constraint_loss
            
            if hasattr(self.policy, "extra_loss"):
                for key, value in extra_loss.items():
                    loss += value

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

            # NOTE: using stablebaseline3 implementation
            num_updates += 1

        # -- For PPO
        # num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_clipping_ratio = (num_clipped_ratios / num_all_ratios) if num_all_ratios > 0 else 0
        mean_kl = mean_kl / num_updates

        if self.desired_clipping > 0.0:
            if self.gpu_global_rank == 0:
                if mean_clipping_ratio > self.desired_clipping * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif mean_clipping_ratio < self.desired_clipping / 2.0 and mean_clipping_ratio > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            # Update the learning rate for all GPUs
            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()

            # Update the learning rate for all parameter groups
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate


        for key, value in mean_extra_loss.items():
            mean_extra_loss[key] /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()
        if hasattr(self.policy, "after_train"):
            self.policy.after_train()
            
                    
        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "clipping_ratio": mean_clipping_ratio,
            "mean_kl": mean_kl,
            **mean_extra_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict