import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from isaaclab_rl.rsl_rl.utils import resolve_nn_activation
from isaaclab_rl.rsl_rl.networks.transformer_discriminator import TransformerDiscriminator

class AmpReward:
    def __init__(self, input_dim: int,
                 learning_rate: float = 1e-4,
                 num_learning_epochs: int = 3,
                 store_interval: int = 8,
                 max_ref_buffer_size: int = 100000,
                 max_gen_buffer_size: int = 100000,
                 sample_k: int = 1024,
                 hidden_dims: list[int] = [128, 128],
                 activation: str = "relu",
                 layer_norm: bool = False,
                 clip_obs_value: float = 100,
                 w_grad_penalty: float = 10.0,
                 max_grad_norm: float = 1.0,
                 reward_scale: float = 1.0,
                 reward_factor: float = 0.0,
                 weight_decay: float = 0.0, 

                 use_transformer: bool = False,
                 tf_d_model: int = 256,
                 tf_hidden_dim: int = 512,
                 tf_num_layers: int = 3,
                 tf_num_heads: int = 4,
                 tf_dropout: float = 0.1,
                 tf_num_input_tokens: int = 4,
                 tf_activation: str = "gelu",

                 training: bool = True,
                 num_envs: int = 1,
                 device: str = "cpu",
                 offload_buffer: bool = False,
                 multi_gpu_cfg: dict | None = None,):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.layer_norm = layer_norm
        self.device = device
        self.learning_rate = learning_rate
        self.num_learning_epochs = num_learning_epochs

        self.num_envs = num_envs
        self.clip_obs_value = clip_obs_value
        self.reward_scale = reward_scale
        self.reward_factor = reward_factor
        self.w_grad_penalty = w_grad_penalty
        self.max_grad_norm = max_grad_norm

        self.store_interval = store_interval
        max_ref_buffer_size = max_ref_buffer_size - max_ref_buffer_size % num_envs
        assert max_ref_buffer_size > 0, "max_buffer_size must be greater than 0 after mod num_envs"
        self.max_ref_buffer_size = max_ref_buffer_size
        max_gen_buffer_size = max_gen_buffer_size - max_gen_buffer_size % num_envs
        assert max_gen_buffer_size > 0, "max_gen_buffer_size must be greater than 0 after mod num_envs"
        assert max_ref_buffer_size % sample_k == 0, "max_ref_buffer_size must be divisible by sample_k"
        assert max_gen_buffer_size % sample_k == 0, "max_gen_buffer_size must be divisible by sample_k"
        sample_k = min(sample_k, num_envs)
        assert num_envs % sample_k == 0, "num_envs must be divisible by sample_k"
        self.max_gen_buffer_size = max_gen_buffer_size
        self.max_ref_buffer_step = max_ref_buffer_size // num_envs
        self.max_gen_buffer_step = max_gen_buffer_size // num_envs
        self.sample_k = sample_k

        self.use_transformer = use_transformer
        self.tf_d_model = tf_d_model
        self.tf_hidden_dim = tf_hidden_dim
        self.tf_num_layers = tf_num_layers
        self.tf_num_heads = tf_num_heads
        self.tf_dropout = tf_dropout
        self.tf_num_input_tokens = tf_num_input_tokens

        self.offload_buffer = offload_buffer
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
            

        activation = resolve_nn_activation(activation)
        # create the network
        if not use_transformer:
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dims[0]))
            layers.append(deepcopy(activation))
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden_dims[i + 1]))
                layers.append(deepcopy(activation))
            layers.append(nn.Linear(hidden_dims[-1], 1))
            # layers.append(nn.Sigmoid()) # NOTE: no sigmoid for amp
            self.network = nn.Sequential(*layers)
            self.network.to(device)
        else:
            tf_activation = resolve_nn_activation(tf_activation)
            self.network = TransformerDiscriminator(
                input_dim,
                1,
                hidden_dims,
                activation,
                tf_num_input_tokens,
                tf_d_model,
                tf_num_layers,
                tf_num_heads,
                tf_hidden_dim,
                tf_dropout,
                tf_activation,
                enable_sdpa=False, # False to compute second order gradient
            )
            self.network.to(device)

        self.training = training
        if training:
            self.step_counter = 0
            self.num_ref_storage = 0
            self.num_gen_storage = 0
            self.gen_storage = torch.zeros(max_gen_buffer_size, input_dim, device=device if not offload_buffer else "cpu")
            self.ref_storage = torch.zeros(max_ref_buffer_size, input_dim, device=device if not offload_buffer else "cpu")
            self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def reset_storage(self):
        if self.training:
            self.step_counter = 0
            self.num_ref_storage = 0
            self.num_gen_storage = 0
            self.gen_storage[:] = 0.
            self.ref_storage[:] = 0.

    def _compute_gradient_penalty(self, real_samples, fake_samples):
        # Randomly interpolate between real and fake samples
        interpolated = real_samples.clone() # NOTE: real only
        interpolated.requires_grad_(True)

        return self._compute_gradient_penalty_(interpolated)

    def _compute_gradient_penalty_(self, interpolated):
        # Compute discriminator output
        d_interpolated = self.network(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                        grad_outputs=torch.ones_like(d_interpolated),
                                        create_graph=True, retain_graph=True)[0]

        # Compute the gradient penalty
        gradient_penalty = gradients.square().sum(dim=-1).mean() * self.w_grad_penalty * 0.5
        return gradient_penalty

    def _optimize_amp(self, gen_batch, ref_batch) -> float:
        score_ref = self.network(ref_batch)
        score_gen = self.network(gen_batch)
        disc_loss = torch.square(score_ref - 1.).mean() + torch.square(score_gen + 1.).mean()
        grad_penalty = self._compute_gradient_penalty(ref_batch, gen_batch)
        amp_loss = disc_loss + grad_penalty

        self.optimizer.zero_grad()
        amp_loss.backward()

        if self.gpu_world_size > 1:
            self.reduce_parameters()

        norm = nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        if torch.any(torch.logical_or(torch.isnan(norm), torch.isinf(norm))):
            print('[WARNING]: AMP Encountered Invalid Gradient, Ignoring')
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return disc_loss.item(), grad_penalty.item()

    def update(self):
        if not self.training:
            raise ValueError("AMP is not trained, call training=True")
        mean_disc_loss, mean_grad_penalty = 0, 0
        num_updates = 0
        
        gen_storage = self.gen_storage[:self.num_gen_storage].clone()
        ref_storage = self.ref_storage[:self.num_ref_storage].clone()
        gen_batch_ids = torch.randperm(self.num_gen_storage, device=self.device if not self.offload_buffer else "cpu")
        ref_batch_ids = torch.randperm(self.num_ref_storage, device=self.device if not self.offload_buffer else "cpu")

        assert self.step_counter > 0, "step_counter must be greater than 0 when updating"
        latest_gen_idx = int((((self.step_counter - 1) // self.store_interval) % self.max_gen_buffer_step) * self.num_envs // self.sample_k)
        latest_ref_idx = int((((self.step_counter - 1) // self.store_interval) % self.max_ref_buffer_step) * self.num_envs // self.sample_k)
        for i in range(self.num_learning_epochs):
            gen_start_idx = ((i + latest_gen_idx) * self.sample_k) % self.num_gen_storage
            gen_end_idx = gen_start_idx + self.sample_k
            ref_start_idx = ((i + latest_ref_idx) * self.sample_k) % self.num_ref_storage
            ref_end_idx = ref_start_idx + self.sample_k

            gen_batch = gen_storage[gen_batch_ids[gen_start_idx:gen_end_idx]]
            ref_batch = ref_storage[ref_batch_ids[ref_start_idx:ref_end_idx]]
            if self.offload_buffer:
                gen_batch = gen_batch.to(self.device)
                ref_batch = ref_batch.to(self.device)
            disc_loss, grad_penalty = self._optimize_amp(gen_batch, ref_batch)
            mean_disc_loss += disc_loss
            mean_grad_penalty += grad_penalty
            num_updates += 1

        mean_disc_loss /= num_updates if num_updates > 0 else torch.tensor(1.0, device=self.device)
        mean_grad_penalty /= num_updates if num_updates > 0 else torch.tensor(1.0, device=self.device)
        return mean_disc_loss, mean_grad_penalty

    @torch.inference_mode()
    def update_storage(self, gen_obs: torch.Tensor, ref_obs: torch.Tensor):
        if self.step_counter % self.store_interval == 0:
            ref_start_idx = ((self.step_counter // self.store_interval) % self.max_ref_buffer_step) * self.num_envs
            gen_start_idx = ((self.step_counter // self.store_interval) % self.max_gen_buffer_step) * self.num_envs
            ref_end_idx = ref_start_idx + self.num_envs
            gen_end_idx = gen_start_idx + self.num_envs
            if self.offload_buffer:
                gen_obs = gen_obs.to("cpu")
                ref_obs = ref_obs.to("cpu")
            self.num_ref_storage = min(self.max_ref_buffer_size, self.num_ref_storage + self.num_envs)
            self.num_gen_storage = min(self.max_gen_buffer_size, self.num_gen_storage + self.num_envs)
            self.ref_storage[ref_start_idx:ref_end_idx, :] = ref_obs.view(self.num_envs, -1).clamp(-self.clip_obs_value, self.clip_obs_value)
            self.gen_storage[gen_start_idx:gen_end_idx, :] = gen_obs.view(self.num_envs, -1).clamp(-self.clip_obs_value, self.clip_obs_value)
        self.step_counter += 1

    @torch.inference_mode()
    def compute_reward(self, obs: torch.Tensor, scale: float | torch.Tensor = 1.0):
        features = obs.view(self.num_envs, -1).clamp(-self.clip_obs_value, self.clip_obs_value)
        amp_score = self.network(features).view(self.num_envs)

        # if scale is low, reward will be high to make amp constraints less important
        return (1 - scale * self.reward_factor * torch.square(amp_score - 1)) * self.reward_scale
    
    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.network.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.network.parameters()

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
    
    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()
