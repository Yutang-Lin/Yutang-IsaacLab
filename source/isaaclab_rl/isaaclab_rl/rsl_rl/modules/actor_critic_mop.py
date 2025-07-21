import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation
from .actor_critic import ActorCritic, ResidualWrapper
from copy import deepcopy

class MoPModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_policies,
                 router_hidden_dims,
                 activation,
                 layer_norm=False,
                 dropout_rate=0.0,
                 residual=False,
                 store_logits=[]):
        super(MoPModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_features = input_dim
        self.out_features = output_dim
        self.num_policies = num_policies
        self.hidden_dims = hidden_dims
        self.store_logits = store_logits
        self.logits = torch.tensor(0.0)
        
        single_model = []
        for i in range(len(hidden_dims)):
            if i == 0:
                single_model.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                single_model.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                if layer_norm:
                    single_model.append(nn.LayerNorm(hidden_dims[i]))
                single_model.append(activation)
                if dropout_rate > 0:
                    single_model.append(nn.Dropout(dropout_rate))
        if residual:
            seq_model = nn.Sequential(*single_model)
            single_model = [ResidualWrapper(seq_model, input_dim, hidden_dims[-1]),
                            nn.Linear(hidden_dims[-1], output_dim)]
        else:
            single_model.append(nn.Linear(hidden_dims[-1], output_dim))
        single_model = nn.Sequential(*single_model)
        self.policies = nn.ModuleList([deepcopy(single_model) for _ in range(num_policies)])
        
        self.router_hidden_dims = router_hidden_dims
        router = []
        for i in range(len(router_hidden_dims)):
            if i == 0:
                router.append(nn.Linear(input_dim, router_hidden_dims[i]))
            else:
                router.append(nn.Linear(router_hidden_dims[i-1], router_hidden_dims[i]))
                if layer_norm:
                    router.append(nn.LayerNorm(router_hidden_dims[i]))
                router.append(activation)
                if dropout_rate > 0:
                    router.append(nn.Dropout(dropout_rate))
        if len(router) == 0:
            router_hidden_dims = [input_dim]
        router.append(nn.Linear(router_hidden_dims[-1], num_policies))
        self.router = nn.Sequential(*router)

    def forward(self, x):
        logits = self.router(x)
        chosen_policies = torch.argmax(logits, dim=-1)

        results = torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)
        for i, policy in enumerate(self.policies):
            chosen_mask = chosen_policies == i
            chosen_batch = chosen_mask.nonzero()

            if chosen_batch.numel() > 0:
                results[chosen_batch] += policy(x[chosen_batch])

        if self.store_logits[0]:
            self.logits = logits
        return results

class ActorCriticMoP(ActorCritic):
    '''Actor Critic with Mixture of Policies'''
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],

        mop_critic=True,
        num_policies=1,
        router_hidden_dims=[128],
        balance_tolerance=0.25,
        balance_loss_weight=2.0,
        grad_penalty_weight=0.0,

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
                "ActorCriticMoP.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation)

        self.store_logits = [False]
        self.balance_tolerance = balance_tolerance
        self.balance_loss_weight = balance_loss_weight  
        self.grad_penalty_weight = grad_penalty_weight
        self.num_policies = num_policies
        self.mop_critic = mop_critic
        self.load_noise_std = load_noise_std

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        self.actor = MoPModule(
            mlp_input_dim_a,
            actor_hidden_dims,
            num_actions,
            num_policies,
            router_hidden_dims,
            activation,
            layer_norm,
            dropout_rate,
            residual,
            self.store_logits,
        )

        # Value function
        if not mop_critic:
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
        else:
            self.critic = MoPModule(
                mlp_input_dim_c,
                critic_hidden_dims,
                1,
                num_policies,
                router_hidden_dims,
                activation,
                layer_norm,
                dropout_rate,
                residual,
                self.store_logits,
            )

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
        mean_prob = 1.0 / self.num_policies
        upper_bound = mean_prob * (1 + self.balance_tolerance)
        lower_bound = mean_prob * (1 - self.balance_tolerance)

        all_balance_loss += self._get_balance_loss(self.actor, upper_bound, lower_bound)
        if self.mop_critic:
            assert isinstance(self.critic, MoPModule), "Critic must be a MoPModule"
            all_balance_loss += self._get_balance_loss(self.critic, upper_bound, lower_bound)

        if self.grad_penalty_weight == 0:
            return {"mop_balance": all_balance_loss * self.balance_loss_weight}
        
        # Actor gradient penalty
        assert 'obs_batch' in kwargs, "obs_batch must be provided when grad_penalty_weight > 0"
        obs_batch: torch.Tensor = kwargs['obs_batch'].clone()
        obs_batch.requires_grad_(True)
        router_logits = self.actor.router(obs_batch)
        nabla_logits = torch.autograd.grad(outputs=router_logits, inputs=obs_batch,
                                          grad_outputs=torch.ones_like(router_logits),
                                          create_graph=True,
                                          allow_unused=True)[0]
        grad_penalty_loss = nabla_logits.square().sum(dim=-1).mean() * 0.5
        if self.mop_critic:
            # Critic gradient penalty
            assert 'critic_obs_batch' in kwargs, "critic_obs_batch must be provided when mop_critic is True"
            critic_obs_batch: torch.Tensor = kwargs['critic_obs_batch'].clone()
            critic_obs_batch.requires_grad_(True)
            critic_logits = self.critic.router(critic_obs_batch)
            nabla_critic_logits = torch.autograd.grad(outputs=critic_logits, inputs=critic_obs_batch,
                                                     grad_outputs=torch.ones_like(critic_logits),
                                                     create_graph=True,
                                                     allow_unused=True)[0]
            grad_penalty_loss += nabla_critic_logits.square().sum(dim=-1).mean() * 0.5

        return {"mop_balance": all_balance_loss * self.balance_loss_weight,
                "mop_grad_penalty": grad_penalty_loss * self.grad_penalty_weight}
        

    def _get_balance_loss(self, module: MoPModule, upper_bound, lower_bound):
        assert module.logits is not None
        activation = torch.softmax(module.logits, dim=-1).mean(dim=0)
        balance_loss = (activation - upper_bound).clamp(min=0.0).sum(dim=-1) + \
                    (lower_bound - activation).clamp(min=0.0).sum(dim=-1)
        module.logits = None
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

