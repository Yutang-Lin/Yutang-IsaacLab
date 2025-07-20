from rsl_rl.modules.actor_critic import ActorCritic as RslRlActorCritic
import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

class ResidualWrapper(nn.Module):
    def __init__(self, module: nn.Module,
                 input_dim: int,
                 output_dim: int):
        super(ResidualWrapper, self).__init__()
        self.module = module
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.residual_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(negative_slope=0.01),
        )
        for layer in self.residual_layer:   
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-0.03, 0.03)
                layer.bias.data.zero_()

    def forward(self, x):
        return self.residual_layer(x) + self.module(x)

class ActorCritic(RslRlActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        load_noise_std: bool = True,
        noise_std_type: str = "scalar",
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        residual: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(RslRlActorCritic, self).__init__()
        activation = resolve_nn_activation(activation)

        self.load_noise_std = load_noise_std

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                if not residual:
                    actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
                else:
                    sequential_actor_layers = nn.Sequential(*actor_layers)
                    residule_wrapper = ResidualWrapper(sequential_actor_layers, mlp_input_dim_a,
                                                       actor_hidden_dims[layer_index])
                    actor_layers = [residule_wrapper, nn.Linear(actor_hidden_dims[layer_index], num_actions)]
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                if layer_norm:
                    actor_layers.append(nn.LayerNorm(actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
                if dropout_rate > 0:
                    actor_layers.append(nn.Dropout(dropout_rate))
        self.actor = nn.Sequential(*actor_layers)

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
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
    
    def extra_loss(self, **kwargs):
        return {}

    def get_actions_log_prob(self, actions, **kwargs):
        return self.distribution.log_prob(actions).sum(dim=-1) # type: ignore

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std + 1e-3) # add small epsilon to avoid log(0)

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
