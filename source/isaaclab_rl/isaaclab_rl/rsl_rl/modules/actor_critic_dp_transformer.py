import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation
from .actor_critic_dp import ActorCriticDP, ResidualWrapper
from isaaclab_rl.rsl_rl.networks.diffusion_transformer import DiffusionTransformer
from isaaclab_rl.rsl_rl.networks.ddim_scheduler import DDIMScheduler

class ActorCriticDPTransformer(ActorCriticDP):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        critic_hidden_dims,
        tf_d_model,
        tf_num_heads,
        tf_hidden_dim,
        tf_num_layers,
        tf_condition_tokens,
        timestep_hidden_dim: int = 256,
        max_timesteps: int = 1000,
        action_timestep: int = 50,
        action_step_num: int = 5,
        diffusion_loss_step_num: int = 2,
        reference_loss_step_num: int = 1,
        reference_gradient: bool = False,
        alphas: torch.Tensor | list[float] | None = None,
        sigmas: torch.Tensor | list[float] | None = None,
        ddim_lambda: float = 1.0,
        ddim_eta: float = 0.0,
        lernable_sigmas: bool = False,
        learn_residual: bool = False,
        activation: str | nn.Module = "elu",
        init_noise_std: float = 1.0,
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
                "ActorCriticDPTransformer.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation) # type: ignore

        self.load_noise_std = load_noise_std
        self.learnable_noise_std = learnable_noise_std
        self.max_timesteps = max_timesteps
        self.action_timestep = action_timestep
        self.action_step_num = action_step_num
        self.reference_gradient = reference_gradient
        self.num_actions = num_actions
        self.ddim_lambda = ddim_lambda
        self.diffusion_loss_step_num = diffusion_loss_step_num
        self.reference_loss_step_num = reference_loss_step_num

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        self.actor = DiffusionTransformer(
            action_dim=num_actions,
            condition_dim=mlp_input_dim_a,
            timestep_dim=timestep_hidden_dim,
            d_model=tf_d_model,
            num_heads=tf_num_heads,
            hidden_dim=tf_hidden_dim,
            num_layers=tf_num_layers,
            condition_tokens=tf_condition_tokens,
            activation=activation,
            dropout=dropout_rate
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

        print(f"Actor Transformer: {self.actor}")
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