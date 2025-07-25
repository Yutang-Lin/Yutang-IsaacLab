import torch
import torch.nn as nn
import wandb
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation
from .actor_critic import ActorCritic, ResidualWrapper
from copy import deepcopy

class Primitive(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation, primitive_id,
                 layer_norm=False,
                 dropout_rate=0.0):
        super(Primitive, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_features = input_dim
        self.out_features = output_dim
        self.hidden_dims = hidden_dims
        self.primitive_id = primitive_id
        
        self.models = nn.ModuleList()
        self.activations = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.models.append(nn.Linear(input_dim, hidden_dims[i]))
                self.activations.append(activation)
            else:
                local_model = []
                local_model.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                if layer_norm:
                    local_model.append(nn.LayerNorm(hidden_dims[i]))
                self.models.append(nn.Sequential(*local_model))

                local_model = []
                local_model.append(activation)
                if dropout_rate > 0:
                    local_model.append(nn.Dropout(dropout_rate))
                self.activations.append(nn.Sequential(*local_model))

        self.models.append(nn.Linear(hidden_dims[-1], output_dim))
        self.activations.append(torch.nn.Identity())

        if self.primitive_id > 0:
            lateral_weights = nn.ModuleList([nn.Linear(hidden_dims[i-1],
                                                        hidden_dims[i]) for i in range(1, len(hidden_dims))])
            self.lateral_weights = nn.ModuleList([deepcopy(lateral_weights) for _ in range(self.primitive_id)])
        else:
            self.lateral_weights = None

    def _forward_single(self, layer_id, x, lateral_input=None):
        x = self.models[layer_id](x)
        if lateral_input is not None and self.lateral_weights is not None:
            for i, l in enumerate(lateral_input):
                # detach the lateral input to avoid gradient flow back to the previous primitive
                x = x + self.lateral_weights[i][layer_id-1](l.detach()) # type: ignore
        return self.activations[layer_id](x)
    
    def _forward_full(self, x, lateral_input=None,
                      drop_last_layers=False):
        all_hiddens = []
        # [None, lateral_1, lateral_2, ..., lateral_n, None]
        if self.primitive_id == 0:
            laterals = [None] * len(self.models)
        else:
            laterals = [None] + list(zip(*lateral_input)) + [None]

        # drop_last_layer is True -> drop the last two layers
        for i in range(len(self.models) - (2 if drop_last_layers else 0)):
            x = self._forward_single(i, x, laterals[i])
            all_hiddens.append(x)
        return all_hiddens

    def forward(self, x, lateral_input=None, return_all=True,
                drop_last_layers=False):
        all_hiddens = self._forward_full(x, lateral_input, drop_last_layers)
        if return_all:
            return all_hiddens
        else:
            assert not drop_last_layers, "drop_last_layers must be False when return_all is False"
            return all_hiddens[-1]

class PNNModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 num_policies,
                 router_hidden_dims,
                 activation,
                 current_policy_id=0,
                 layer_norm=False,
                 dropout_rate=0.0,
                 weight_sharing=True):
        super(PNNModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_features = input_dim
        self.out_features = output_dim
        self.num_policies = num_policies
        self.current_policy_id = current_policy_id
        self.weight_sharing = weight_sharing
        self.hidden_dims = hidden_dims
        
        self.policies = nn.ModuleList([Primitive(input_dim, hidden_dims, output_dim,
                                                activation,
                                                id,
                                                layer_norm,
                                                dropout_rate) for id in range(num_policies)])
        
        self.router_hidden_dims = router_hidden_dims
        router = []
        for i in range(len(router_hidden_dims)):
            if i == 0:
                router.append(nn.Linear(input_dim, router_hidden_dims[i]))
                router.append(activation)
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

    def set_policy_id(self, policy_id):
        self.current_policy_id = policy_id

    def increase_policy_id(self, share_weights=True):
        self.current_policy_id += 1
        if self.current_policy_id >= self.num_policies:
            self.current_policy_id = 0
            return False
        elif self.weight_sharing and share_weights:
            last_policy_id = self.current_policy_id - 1
            assert last_policy_id >= 0, "RuntimeError: current_policy_id is 0 but weight_sharing is True"
            state_dict = self.policies[last_policy_id].state_dict()
            model_dict = {k: v for k, v in state_dict.items() if k not in ['lateral_weights']}
            # we only load main model parameters, not lateral weights
            self.policies[self.current_policy_id].load_state_dict(model_dict, strict=False)
            print(f'[INFO]: PNN next policy loaded with weight sharing: {self.current_policy_id+1}/{self.num_policies}', flush=True)
        return True

    def forward(self, x):
        laterals = []
        for i in range(self.current_policy_id + 1):
            all_hiddens = self.policies[i].forward(x, laterals,
                                                   drop_last_layers=(i != self.current_policy_id))
            laterals.append(all_hiddens)
        return laterals[-1][-1]

class ActorCriticPNN(ActorCritic):
    '''Actor Critic with Progressive Neural Network'''
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],

        num_policies=1,
        pnn_critic=False,
        weight_sharing=True,
        start_by_id=0,
        router_hidden_dims=[128],
        grad_penalty_weight=0.0,

        activation="elu",
        init_noise_std=1.0,
        load_noise_std: bool = True,
        learnable_noise_std: bool = True,
        noise_std_type: str = "scalar",
        layer_norm: bool = False,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMoP.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        nn.Module.__init__(self)
        activation = resolve_nn_activation(activation)
        assert len(actor_hidden_dims) > 1, "actor_hidden_dims must have at least 2 layers"
        if pnn_critic:
            assert len(critic_hidden_dims) > 1, "critic_hidden_dims must have at least 2 layers"

        self.grad_penalty_weight = grad_penalty_weight
        self.num_policies = num_policies
        self.pnn_critic = pnn_critic
        self.weight_sharing = weight_sharing
        self.start_by_id = start_by_id
        self.load_noise_std = load_noise_std
        self.learnable_noise_std = learnable_noise_std

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        self.actor = PNNModule(
            mlp_input_dim_a,
            actor_hidden_dims,
            num_actions,
            num_policies,
            router_hidden_dims,
            activation,
            0,
            layer_norm,
            dropout_rate,
            weight_sharing,
        )

        # Value function
        if pnn_critic:
            self.critic = PNNModule(
                mlp_input_dim_c,
                critic_hidden_dims,
                1,
                num_policies,
                router_hidden_dims,
                activation,
                0,
                layer_norm,
                dropout_rate,
                weight_sharing,
            )
        else:
            critic = []
            critic_hidden_dims = [mlp_input_dim_c] + critic_hidden_dims
            for i in range(len(critic_hidden_dims) - 1):
                critic.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i+1]))
                if layer_norm:
                    critic.append(nn.LayerNorm(critic_hidden_dims[i+1]))
                critic.append(activation)
                if dropout_rate > 0:
                    critic.append(nn.Dropout(dropout_rate))
            critic.append(nn.Linear(critic_hidden_dims[-1], 1))
            self.critic = nn.Sequential(*critic)

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
        device = self.std.device if hasattr(self, 'std') else self.log_std.device
        loss_dict = {'schedule': torch.tensor(self.actor.current_policy_id, device=device)}
        if self.grad_penalty_weight == 0:
            return loss_dict
        
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

        # Critic gradient penalty
        if self.pnn_critic:
            assert 'critic_obs_batch' in kwargs, "critic_obs_batch must be provided when grad_penalty_weight > 0"
            critic_obs_batch: torch.Tensor = kwargs['critic_obs_batch'].clone()
            critic_obs_batch.requires_grad_(True)
            critic_logits = self.critic.router(critic_obs_batch)
            nabla_critic_logits = torch.autograd.grad(outputs=critic_logits, inputs=critic_obs_batch,
                                                        grad_outputs=torch.ones_like(critic_logits),
                                                        create_graph=True,
                                                        allow_unused=True)[0]
            grad_penalty_loss += nabla_critic_logits.square().sum(dim=-1).mean() * 0.5

        loss_dict["pnn_grad_penalty"] = grad_penalty_loss * self.grad_penalty_weight
        return loss_dict
    
    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean
    
    def schedule(self, converged) -> dict:
        return_dict = dict(
            num_policies=self.num_policies,
            current_policy_id=self.actor.current_policy_id,
            rescheduled=False)
        if converged:
            rescheduled = not self.actor.increase_policy_id()
            if self.pnn_critic:
                self.critic.increase_policy_id()
            if wandb.run is not None:
                wandb.alert(title="PNN schedule alert", 
                            text=f'Current PNN schedule: {self.actor.current_policy_id+1}/{self.num_policies}')
                if rescheduled:
                    wandb.alert(title="PNN schedule alert", 
                                text=f'PNN rescheduled: {self.actor.current_policy_id+1}/{self.num_policies}')
            
            print(f'[INFO]: PNN schedule: {self.actor.current_policy_id+1}/{self.num_policies}', flush=True)
            return_dict['rescheduled'] = rescheduled
            return_dict['current_policy_id'] = self.actor.current_policy_id
        
        return return_dict
        
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

        # pnn load unstrictly to support changing the number of policies
        try:
            super().load_state_dict(state_dict, strict=False)
        except Exception as e:
            # this due to mismatch of the number of policies
            # thus delete router from state_dict
            state_dict = {k: v for k, v in state_dict.items() if 'router' not in k}
            super().load_state_dict(state_dict, strict=False)
            print(f'[WARNING]: PNN load state_dict with mismatch number of policies, initializing new router.', flush=True)

        for _ in range(self.start_by_id):
            self.actor.increase_policy_id(share_weights=False)
            if self.pnn_critic:
                self.critic.increase_policy_id(share_weights=False)
        return True