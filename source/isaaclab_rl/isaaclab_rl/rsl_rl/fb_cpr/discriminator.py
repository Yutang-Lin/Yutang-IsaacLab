import torch
import torch.nn as nn
from ..networks.transformer import MLP
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, dim_state, dim_action, dim_latent,
                 grad_penalty_weight: float = 10.0):
        super(Discriminator, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_latent = dim_latent
        self.grad_penalty_weight = grad_penalty_weight

        self.discriminator = MLP(
            input_dim=dim_state + dim_latent,
            hidden_dims=[1024, 512],
            output_dim=1,
            activation=nn.Mish(),
        )

    def forward(self, state, latent):
        inputs = torch.cat([state, latent], dim=-1)
        return self.discriminator(inputs)
    
    def compute_reward(self, state, latent, eps: float = 1e-7):
        outputs = self(state, latent)
        outputs = torch.clamp(outputs, min=eps, max=1-eps)
        rewards = outputs.log() - (1 - outputs).log()
        return rewards
    
    def loss(self, real_state, real_latent, fake_state, fake_latent):
        real_outputs = self(real_state, real_latent)
        fake_outputs = self(fake_state, fake_latent)
        loss = torch.mean(-F.logsigmoid(real_outputs) + F.logsigmoid(-fake_outputs))
        return loss + (
                        self.grad_penalty_weight * \
                        self.gradient_penalty_wgan(real_state, real_latent, fake_state, fake_latent)
                    )
    
    @torch.compiler.disable
    def gradient_penalty_wgan(
        self,
        real_state: torch.Tensor,
        real_latent: torch.Tensor,
        fake_state: torch.Tensor,
        fake_latent: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = real_state.shape[0]
        alpha = torch.rand(batch_size, 1, device=real_state.device)
        interpolates = torch.cat(
            [
                (alpha * real_state + (1 - alpha) * fake_state).requires_grad_(True),
                (alpha * real_latent + (1 - alpha) * fake_latent).requires_grad_(True),
            ],
            dim=1,
        )
        d_interpolates = self.discriminator(
            interpolates[:, 0 : real_state.shape[1]], interpolates[:, real_state.shape[1] :]
        )
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty