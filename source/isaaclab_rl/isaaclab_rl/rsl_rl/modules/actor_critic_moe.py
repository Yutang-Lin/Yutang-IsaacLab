import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation
from .actor_critic import ActorCritic, ResidualWrapper

class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k, store_logits: list[bool]):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.store_logits = store_logits
        self.logits = torch.tensor(0.0)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_features = input_dim
        self.out_features = output_dim
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.router(x)
        chosen_experts = torch.topk(logits, self.top_k, dim=-1)
        chosen_experts, weight = chosen_experts.indices, chosen_experts.values
        weight = weight.softmax(dim=-1)

        results = torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
        for i, expert in enumerate(self.experts):
            chosen_mask = chosen_experts == i
            chosen_batch = torch.any(chosen_experts == i, dim=-1).nonzero()

            if chosen_batch.numel() > 0:
                chosen_weight = (weight * chosen_mask).sum(dim=-1, keepdim=True)
                results[chosen_batch] += expert(x[chosen_batch]) * chosen_weight[chosen_batch]

        if self.store_logits[0]:
            self.logits = logits
        return results

class ActorCriticMoE(ActorCritic):
    '''Actor Critic with Mixture of Experts'''
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],

        moe_critic=False,
        num_experts=1,
        top_k=1,
        balance_tolerance=0.25,
        balance_loss_weight=2.0,

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
                "ActorCriticMoE.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation)

        self.store_logits = [False]
        self.balance_tolerance = balance_tolerance
        self.balance_loss_weight = balance_loss_weight  
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_critic = moe_critic
        self.load_noise_std = load_noise_std
        self.learnable_noise_std = learnable_noise_std

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(MoELayer(mlp_input_dim_a, actor_hidden_dims[0], num_experts, top_k, self.store_logits))
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
                actor_layers.append(MoELayer(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1], num_experts, top_k, self.store_logits))
                if layer_norm:
                    actor_layers.append(nn.LayerNorm(actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
                if dropout_rate > 0:
                    actor_layers.append(nn.Dropout(dropout_rate))
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        if self.moe_critic:
            critic_layers.append(MoELayer(mlp_input_dim_c, critic_hidden_dims[0], num_experts, top_k, self.store_logits))
        else:
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
                if self.moe_critic:
                    critic_layers.append(MoELayer(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1], num_experts, top_k, self.store_logits))
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
        self.distribution: Normal = None # type: ignore
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def extra_loss(self, **kwargs):
        all_balance_loss = 0.0
        mean_prob = 1.0 / self.num_experts
        upper_bound = mean_prob * (1 + self.balance_tolerance)
        lower_bound = mean_prob * (1 - self.balance_tolerance)

        for layer in self.actor:
            if isinstance(layer, MoELayer):
                all_balance_loss += self._get_balance_loss(layer, upper_bound, lower_bound)
            elif isinstance(layer, ResidualWrapper):
                for sub_layer in layer.module:
                    if isinstance(sub_layer, MoELayer):
                        all_balance_loss += self._get_balance_loss(sub_layer, upper_bound, lower_bound)
    
        if self.moe_critic:
            for layer in self.critic:
                if isinstance(layer, MoELayer):
                    all_balance_loss += self._get_balance_loss(layer, upper_bound, lower_bound)
                elif isinstance(layer, ResidualWrapper):
                    for sub_layer in layer.module:
                        if isinstance(sub_layer, MoELayer):
                            all_balance_loss += self._get_balance_loss(sub_layer, upper_bound, lower_bound)

        return {"moe_balance": all_balance_loss * self.balance_loss_weight}
    
    def _get_balance_loss(self, layer: MoELayer, upper_bound, lower_bound):
        assert layer.logits is not None
        activation = torch.softmax(layer.logits, dim=-1).mean(dim=0)
        balance_loss = (activation - upper_bound).clamp(min=0.0).sum(dim=-1) + \
                    (lower_bound - activation).clamp(min=0.0).sum(dim=-1)
        layer.logits = None
        return balance_loss
    
    def set_store_logits(self, store_logits):
        self.store_logits[0] = store_logits
    
    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean
    
    def pre_train(self):
        self.set_store_logits(True)

    def after_train(self):
        self.set_store_logits(False)

