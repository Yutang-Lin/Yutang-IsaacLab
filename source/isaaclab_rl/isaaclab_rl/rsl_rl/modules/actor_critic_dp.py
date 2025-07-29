import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation
from isaaclab_rl.rsl_rl.modules.actor_critic import ActorCritic, ResidualWrapper
from isaaclab_rl.rsl_rl.networks.diffusion_mlp import DiffusionMLP
from isaaclab_rl.rsl_rl.networks.ddim_scheduler import DDIMScheduler

class ActorCriticDP(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        condition_hidden_dim=256,
        timestep_hidden_dim=256,
        max_timesteps=1000,
        action_timestep=50,
        action_step_num=5,
        reference_gradient=False,
        alphas=None,
        sigmas=None,
        ddim_lambda=1.0,
        ddim_eta=0.0,
        lernable_sigmas=False,
        learn_residual=False,
        activation="elu",
        init_noise_std=1.0,
        load_noise_std: bool = True,
        learnable_noise_std: bool = True,
        noise_std_type: str = "scalar",
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        residual: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDP.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation)

        self.load_noise_std = load_noise_std
        self.learnable_noise_std = learnable_noise_std
        self.max_timesteps = max_timesteps
        self.action_timestep = action_timestep
        self.action_step_num = action_step_num
        self.reference_gradient = reference_gradient
        self.num_actions = num_actions
        self.ddim_lambda = ddim_lambda

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        self.actor = DiffusionMLP(
            action_dim=num_actions,
            condition_dim=mlp_input_dim_a,
            hidden_dims=actor_hidden_dims,
            condition_hidden_dim=condition_hidden_dim,
            timestep_hidden_dim=timestep_hidden_dim,
            activation=activation,
        )
        self.scheduler = DDIMScheduler(
            timestep_hidden_dim=timestep_hidden_dim,
            max_timesteps=max_timesteps,
            alphas=alphas,
            sigmas=sigmas,
            ddim_eta=ddim_eta,
            learnable_sigmas=lernable_sigmas,
            learn_residual=learn_residual,
        )
        self.scheduler.sigmas[self.action_timestep + 1:] = 0.0
        self.scheduler.set_model(self.actor)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                if not residual:
                    critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
                else:
                    sequential_critic_layers = nn.Sequential(*critic_layers)
                    residule_wrapper = ResidualWrapper(sequential_critic_layers, mlp_input_dim_c,
                                                       critic_hidden_dims[layer_index])
                    critic_layers = [residule_wrapper, nn.Linear(critic_hidden_dims[layer_index], 1)]
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                if layer_norm:
                    critic_layers.append(nn.LayerNorm(critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
                if dropout_rate > 0:
                    critic_layers.append(nn.Dropout(dropout_rate))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None # type: ignore
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
    
    def extra_loss(self, 
                   obs_batch,
                   initial_noise_batch,
                   reference_actions_batch,
                   **kwargs):
        current_actions = self.distribution.sample()
        if not self.reference_gradient:
            current_actions = current_actions.detach()
        zero_condition = torch.zeros_like(obs_batch)
        _, pred_reference = self.scheduler.compute_noise_and_x_0_pred(current_actions,
                                                                   obs_batch,
                                                                   timestep=self.action_timestep,
                                                                   condition_lambda=self.ddim_lambda,
                                                                   condition_empty=True)
        reference_loss = (pred_reference - reference_actions_batch).square().mean()

        diffusion_loss = self.scheduler.loss(reference_actions_batch,
                                             zero_condition)
        return {'reference_loss': reference_loss, 'diffusion_loss': diffusion_loss}
    
    def generate_noise(self, num_samples, device):
        return torch.randn(num_samples, self.num_actions, device=device)

    def get_actions_log_prob(self, actions, **kwargs):
        return self.distribution.log_prob(actions).sum(dim=-1) # type: ignore

    def update_distribution(self, observations, noise=None):
        # compute mean
        if noise is None:
            noise = torch.randn(observations.shape[0], self.num_actions, device=observations.device)
        action_prev: torch.Tensor = self.scheduler.solve_grouped(noise, observations, deterministic=True,
                                                    from_timestep=self.max_timesteps - 1,
                                                    to_timestep=self.action_timestep,
                                                    num_steps=self.action_step_num,
                                                    randomize_num_steps=True,
                                                    condition_lambda=self.ddim_lambda) # type: ignore
        
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(action_prev)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(action_prev)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        if not self.learnable_noise_std:
            std = std.detach()
        
        # create distribution
        self.distribution: torch.distributions.Normal = self.scheduler.sample(action_prev, observations,
                                                                    from_timestep=self.action_timestep,
                                                                    condition_lambda=self.ddim_lambda,
                                                                    return_distribution=True,
                                                                    apply_noise=True,
                                                                    sigma_coeff=std) # type: ignore

    def act(self, observations, noise=None, **kwargs):
        self.update_distribution(observations, noise)
        return self.distribution.sample()

    def act_inference(self, observations, noise=None):
        if noise is None:
            noise = torch.zeros(observations.shape[0], self.num_actions, device=observations.device)
        actions_mean: torch.Tensor = self.scheduler.solve_grouped(noise, observations, deterministic=True,
                                                                from_timestep=self.max_timesteps - 1,
                                                                to_timestep=self.action_timestep - 1,
                                                                num_steps=self.action_step_num + 1,
                                                                randomize_num_steps=False,
                                                                condition_lambda=self.ddim_lambda) # type: ignore
        return actions_mean

    def pre_train(self):
        pass

    def after_train(self):
        pass

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        if hasattr(self, "std") and not self.load_noise_std and 'std' in state_dict:
            del state_dict["std"]
            super().load_state_dict(state_dict, strict=False)
            print('[WARNING]: Ignoring std in state_dict, setting strict to False')
            return True
        
        elif hasattr(self, "log_std") and not self.load_noise_std and 'log_std' in state_dict:
            del state_dict["log_std"]
            super().load_state_dict(state_dict, strict=False)
            print('[WARNING]: Ignoring log_std in state_dict, setting strict to False')
            return True

        super().load_state_dict(state_dict, strict=strict)
        return True
