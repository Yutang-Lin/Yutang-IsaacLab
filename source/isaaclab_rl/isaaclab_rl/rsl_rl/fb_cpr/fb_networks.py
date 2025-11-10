import torch
import torch.nn as nn
import torch.nn.functional as F
from ..networks.transformer_policy import TransformerPolicy
from typing import List

class FNetwork(nn.Module):
    def __init__(self, dim_state, dim_action, dim_latent):
        super(FNetwork, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_latent = dim_latent

        self.net = TransformerPolicy(
            input_size=dim_state + dim_action + dim_latent,
            output_size=dim_latent,
            mlp_hidden_dims=[128],
            mlp_activation=nn.Mish(),
            num_input_tokens=2,
            d_model=256,
            num_layers=2,
            num_heads=4,
            hidden_dim=512,
            dropout=0.0,
            activation=nn.GELU(approximate="tanh"),
        )

    def forward(self, state, action, latent):
        x = torch.cat([state, action, latent], dim=1)
        return self.net(x)
    
class BNetwork(nn.Module):
    def __init__(self, dim_state, dim_latent):
        super(BNetwork, self).__init__()
        self.dim_state = dim_state
        self.dim_latent = dim_latent

        self.net = TransformerPolicy(
            input_size=dim_state,
            output_size=dim_latent,
            mlp_hidden_dims=[128],
            mlp_activation=nn.Mish(),
            num_input_tokens=2,
            d_model=256,
            num_layers=2,
            num_heads=4,
            hidden_dim=512,
            dropout=0.0,
            activation=nn.GELU(approximate="tanh"),
        )

    def forward(self, state):
        return self.net(torch.cat([state], dim=1))
    
class FBNetwork(nn.Module):
    def __init__(self, dim_state, dim_action, dim_latent,
                 num_parallel_f_nets: int = 2,
                 gamma: float = 0.98,
                 ortho_loss_weight: float = 100.0):
        super(FBNetwork, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_latent = dim_latent
        self.gamma = gamma
        self.ortho_loss_weight = ortho_loss_weight

        self.f_nets = nn.ModuleList([FNetwork(dim_state, dim_action, dim_latent) 
                                     for _ in range(num_parallel_f_nets)])
        self.b_net = BNetwork(dim_state, dim_latent)

    @torch.no_grad()
    def _compute_td_target(self,
                           batched_next_state: torch.Tensor,
                           batched_next_action: torch.Tensor,
                           batched_latent: torch.Tensor):
        Fs = [f_net(batched_next_state, batched_next_action, batched_latent) for f_net in self.f_nets]
        return (torch.stack(
            [torch.einsum("ik,jk->ij", 
                         f, 
                         self.b_net(batched_next_state)
                        ) for f in Fs]
            , dim=0).detach(), 
            (torch.stack(Fs, dim=0) * batched_latent.unsqueeze(0)).sum(dim=-1).min(dim=0).values.detach()
        )

    def _loss_single_f_net(self, f_net: FNetwork,
                           b_next_state: torch.Tensor,
                           b_cov_target: torch.Tensor,
                           b_z_target: torch.Tensor,
                           td_targets: torch.Tensor,
                           batched_state: torch.Tensor,
                           batched_action: torch.Tensor,
                           batched_latent: torch.Tensor):
        batch_size = batched_state.shape[0]
        Fs = f_net(batched_state, batched_action, batched_latent)
        FB_result = torch.einsum("ik,jk->ij", 
                                 Fs, 
                                 b_next_state
                                )
        Fz_result = (Fs * batched_latent).sum(dim=-1)
        
        diag_mask = 1 - torch.eye(batch_size, device=td_targets.device)
        fb_loss = (batch_size / (2 * batch_size - 2)) * \
                F.mse_loss(FB_result, self.gamma * td_targets.mean(dim=0) * diag_mask) - \
                (1 / batch_size) * FB_result.diag().sum()
        fz_loss = F.mse_loss(Fz_result, b_cov_target.detach() + self.gamma * b_z_target)
        return dict(
            fb_loss=fb_loss,
            fz_loss=fz_loss,
        )

    def loss(self, 
             batched_state: torch.Tensor, 
             batched_next_state: torch.Tensor,
             batched_action: torch.Tensor, 
             batched_next_action: torch.Tensor,
             batched_latent: torch.Tensor):
        b_state = self.b_net(batched_state)
        b_next_state = self.b_net(batched_next_state)
        with torch.no_grad():
            td_targets, b_z_target = self._compute_td_target(
                batched_next_state, 
                batched_next_action, 
                batched_latent
            )
            b_cov_target = b_state.T @ b_state / b_state.shape[0]
            b_cov_target = torch.linalg.solve(b_cov_target, b_next_state, left=False)
            b_cov_target = (b_cov_target * batched_latent).sum(dim=-1)

        fb_losses = [self._loss_single_f_net(
                            f_net, # type: ignore
                            b_next_state, 
                            b_cov_target,
                            b_z_target,
                            td_targets,
                            batched_state,
                            batched_action,
                            batched_latent)
                     for f_net in self.f_nets]
        
        batch_size = b_next_state.shape[0]
        diag_mask = 1 - torch.eye(batch_size, device=b_next_state.device)
        b_inner_product = b_next_state @ b_next_state.T
        ortho_loss = (batch_size / (2 * batch_size - 2)) * \
                ((b_inner_product) * diag_mask).square().mean() - \
                (1 / batch_size) * b_inner_product.diag().sum()
        ortho_loss = ortho_loss * self.ortho_loss_weight
        
        loss_dict = dict(
            fb_loss=torch.tensor(0.0, device=ortho_loss.device),
            fz_loss=torch.tensor(0.0, device=ortho_loss.device),
            ortho_loss=ortho_loss,
        )
        for item in fb_losses:
            for key, value in item.items():
                loss_dict[key] += value # type: ignore
        return loss_dict
    
    def as_q_function(self, state, action, latent):
        Qs = [(f_net(state, action, latent) * latent).sum(dim=-1) for f_net in self.f_nets]
        return torch.stack(Qs, dim=0).min(dim=0).values
    
    @torch.no_grad()
    def encode_trajectory(self, states):
        # states: (batch_size, seq_length, dim_state)
        if states.ndim == 2:
            states = states.unsqueeze(1)
        B, L, D = states.shape
        states = states.view(B * L, D)
        states = self.b_net(states)
        states = states.view(B, L, D).mean(dim=1)
        return states

def test_fb_networks():
    networks = FBNetwork(dim_state=512, dim_action=29, dim_latent=256).cuda()
    optimizer = torch.optim.Adam(networks.parameters(), lr=1e-3)
    for i in range(1000):
        state = torch.randn(1024, 512).cuda()
        next_state = torch.randn(1024, 512).cuda()
        action = torch.randn(1024, 29).cuda()
        next_action = torch.randn(1024, 29).cuda()
        latent = torch.randn(1024, 256).cuda()
        latent = latent / latent.norm(dim=-1, keepdim=True)
        loss = networks.loss(state, next_state, action, next_action, latent)
        total_loss = torch.tensor(0.0, device=state.device)
        for key, value in loss.items():
            total_loss += value
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(loss)