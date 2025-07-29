import torch
import torch.nn as nn
import numpy as np

class DDIMScheduler(nn.Module):
    def __init__(self, timestep_hidden_dim,
                 max_timesteps: int,
                 alphas: float | list[float] | torch.Tensor | None = None,
                 sigmas: float | list[float] | torch.Tensor | None = None,
                 ddim_eta: float = 0.0,
                 learnable_sigmas: bool = False,
                 learn_residual: bool = False):
        super().__init__()
        max_timesteps = max_timesteps + 1

        self.model: list[nn.Module] = None # type: ignore
        self.max_timesteps = max_timesteps
        self.time_embed = nn.Embedding(max_timesteps, timestep_hidden_dim)

        self.learnable_sigmas = learnable_sigmas
        self.learn_residual = learn_residual
        self.ddim_eta = ddim_eta
        self.setup_diffusion_schedule(max_timesteps, alphas, sigmas)

    def setup_diffusion_schedule(self, max_timesteps, alphas, sigmas):
        # basic diffusion schedule
        if alphas is not None:
            if isinstance(alphas, float):
                alphas = torch.ones(max_timesteps) * alphas
            elif isinstance(alphas, list):
                alphas = torch.tensor(alphas)
                assert alphas.ndim == 1 and alphas.shape[0] == max_timesteps
            elif isinstance(alphas, torch.Tensor):
                alphas = alphas.view(-1)
                assert alphas.ndim == 1 and alphas.shape[0] == max_timesteps
            else:
                raise ValueError(f"Invalid alphas type: {type(alphas)}")
        else:
            alphas = 1 - torch.linspace(0.0001, 0.02, max_timesteps)
        
        alpha_bar = torch.cumprod(alphas, dim=-1)
        one_minus_alpha_bar = 1.0 - alpha_bar
        alpha_bar_shifted = torch.cat([torch.ones(1), alpha_bar[:-1]], dim=-1)
        one_minus_alpha_bar_shifted = torch.cat([torch.ones(1), one_minus_alpha_bar[:-1]], dim=-1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(one_minus_alpha_bar)
        sqrt_alpha_bar_shifted = torch.sqrt(alpha_bar_shifted)
        sqrt_one_minus_alpha_bar_shifted = torch.sqrt(one_minus_alpha_bar_shifted)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('one_minus_alpha_bar', one_minus_alpha_bar)
        self.register_buffer('alpha_bar_shifted', alpha_bar_shifted)
        self.register_buffer('one_minus_alpha_bar_shifted', one_minus_alpha_bar_shifted)
        self.register_buffer('sqrt_alpha_bar', sqrt_alpha_bar)
        self.register_buffer('sqrt_one_minus_alpha_bar', sqrt_one_minus_alpha_bar)
        self.register_buffer('sqrt_alpha_bar_shifted', sqrt_alpha_bar_shifted)
        self.register_buffer('sqrt_one_minus_alpha_bar_shifted', sqrt_one_minus_alpha_bar_shifted)
        self.register_buffer('coeff_noise', sqrt_one_minus_alpha_bar / sqrt_alpha_bar)
        self.register_buffer('coeff_mean', 1 / sqrt_alpha_bar)

        self.alphas: torch.Tensor
        self.alpha_bar: torch.Tensor
        self.one_minus_alpha_bar: torch.Tensor
        self.alpha_bar_shifted: torch.Tensor
        self.one_minus_alpha_bar_shifted: torch.Tensor
        self.sqrt_alpha_bar: torch.Tensor
        self.sqrt_one_minus_alpha_bar: torch.Tensor
        self.sqrt_alpha_bar_shifted: torch.Tensor
        self.sqrt_one_minus_alpha_bar_shifted: torch.Tensor
        self.coeff_noise: torch.Tensor
        self.coeff_mean: torch.Tensor

        # DDIM sigma calculation
        if sigmas is not None:
            if isinstance(sigmas, float):
                sigmas = torch.ones(max_timesteps) * sigmas
            elif isinstance(sigmas, list):
                sigmas = torch.tensor(sigmas)
                assert sigmas.ndim == 1 and sigmas.shape[0] == max_timesteps
            elif isinstance(sigmas, torch.Tensor):
                sigmas = sigmas.view(-1)
                assert sigmas.ndim == 1 and sigmas.shape[0] == max_timesteps
            else:
                raise ValueError(f"Invalid sigmas type: {type(sigmas)}")
        else:
            # Calculate DDIM sigmas
            sigmas = torch.zeros(max_timesteps)
            for i in range(max_timesteps):
                if i == max_timesteps - 1:
                    sigmas[i] = 0.0
                else:
                    # DDIM sigma formula: σ_t = η * sqrt((1 - α_{t-1}) / (1 - α_t)) * sqrt(1 - α_t / α_{t-1})
                    if i == 0:
                        sigmas[i] = self.ddim_eta * sqrt_one_minus_alpha_bar[i]
                    else:
                        sigmas[i] = self.ddim_eta * torch.sqrt(
                            (1 - alpha_bar[i-1]) / (1 - alpha_bar[i]) * (1 - alpha_bar[i] / alpha_bar[i-1])
                        )

        if self.learnable_sigmas:
            self.sigmas = nn.Parameter(sigmas)
        else:
            self.register_buffer('sigmas', sigmas)
            self.sigmas: torch.Tensor

    def set_model(self, model):
        self.model = []
        self.model.append(model)

    def forward(self, x, condition, timestep):
        timestep_embed = self.time_embed(timestep)
        return self.model[0](x, condition, timestep_embed)
    
    def apply_noise(self, x, timestep):
        return self.sqrt_alpha_bar[timestep, None] * x + self.sqrt_one_minus_alpha_bar[timestep, None] * torch.randn_like(x)
    
    def compute_noise_and_x_0_pred(self, x, condition, timestep):
        if isinstance(timestep, int):
            timestep = torch.ones(x.shape[0], device=x.device, dtype=torch.long) * timestep
        noise_pred = self.forward(x, condition, timestep)
        if self.learn_residual:
            x_0_pred = (x - self.sqrt_one_minus_alpha_bar[timestep, None] * noise_pred) / self.sqrt_alpha_bar[timestep, None]
        else:
            x_0_pred = noise_pred
            noise_pred = (1 / self.sqrt_one_minus_alpha_bar[timestep, None]) * (x - self.sqrt_alpha_bar[timestep, None] * x_0_pred)
        return noise_pred, x_0_pred
    
    def sample(self, x, condition, from_timestep, to_timestep=None,
               apply_noise: bool = True, sigma_coeff: torch.Tensor | float = 1.0,
               return_distribution: bool = False) -> torch.Tensor | torch.distributions.Normal:
        if isinstance(x, torch.distributions.Normal):
            x = x.sample()
        if isinstance(from_timestep, int):
            from_timestep = torch.ones(x.shape[0], device=x.device, dtype=torch.long) * from_timestep
        
        # Predict noise and x_0 prediction
        noise_pred, x_0_pred = self.compute_noise_and_x_0_pred(x, condition, from_timestep)

        # If to_timestep is not provided, use the previous timestep
        if to_timestep is None:
            to_timestep = (from_timestep - 1).clamp(min=0)
        elif isinstance(to_timestep, int):
            to_timestep = torch.ones(x.shape[0], device=x.device, dtype=torch.long) * to_timestep

        # DDIM sampling formula
        # x_{t-1} = sqrt(α_{t-1}) * x_0 + sqrt(1 - α_{t-1} - σ_t^2) * ε_t + σ_t * z
        mean_coeff = self.sqrt_alpha_bar[to_timestep, None]
        # clip sigmas to max value of sqrt(1 - α_{t-1})
        clamped_sigmas = (self.sigmas[from_timestep, None] * sigma_coeff).clamp(
            max=self.sqrt_one_minus_alpha_bar[to_timestep, None] - 1e-6
        )

        # compute noise coefficient
        noise_coeff = torch.sqrt((
            1 - self.alpha_bar[to_timestep, None] - clamped_sigmas.square()
        ).clamp(min=0.0)) # clamp to prevent negative values

        # Use predicted noise for deterministic part
        x_t_pred = mean_coeff * x_0_pred + noise_coeff * noise_pred

        # Add stochastic noise if requested
        if apply_noise and not return_distribution:
            x_t_pred = x_t_pred + clamped_sigmas * torch.randn_like(x)
        elif apply_noise and return_distribution:
            return torch.distributions.Normal(x_t_pred, clamped_sigmas.expand_as(x_t_pred).clamp(min=1e-6))
        elif return_distribution:
            raise ValueError("return_distribution is valid only when apply_noise is True")
        
        return x_t_pred
    
    def solve_grouped(self, x: torch.Tensor | torch.distributions.Normal, 
                            condition: torch.Tensor,
                            deterministic: bool,
                            from_timestep: int, # must solve with batched of same timestep
                            to_timestep: int,
                            num_steps: int,
                            randomize_num_steps: bool = False,
                            sigma_coeff: torch.Tensor | float = 1.0,
                            return_distribution: bool = False) -> torch.Tensor | torch.distributions.Normal:
        timesteps = torch.linspace(from_timestep, to_timestep, num_steps).long().tolist()
        if randomize_num_steps:
            new_timesteps = []
            new_timesteps.append(torch.ones(condition.shape[0], device=condition.device, dtype=torch.long) * \
                                            timesteps[0])
            for cur_step, nxt_step in zip(timesteps[:-1], timesteps[1:]):
                new_timesteps.append((torch.rand(condition.shape[0], device=condition.device) * \
                                       (nxt_step - cur_step) + cur_step).clamp(max=nxt_step - 0.5).long())
            new_timesteps.append(torch.ones(condition.shape[0], device=condition.device, dtype=torch.long) * \
                                            timesteps[-1])
            timesteps = new_timesteps

        for cur_step, nxt_step in zip(timesteps[:-1], timesteps[1:]):
            # sampling
            x = self.sample(x, condition,
                            cur_step, nxt_step, 
                            apply_noise=not deterministic, 
                            sigma_coeff=sigma_coeff, 
                            return_distribution=return_distribution)
        return x
    
    def loss(self, x, condition):
        # Randomly sample timesteps for each data point in the batch (excluding t=0)
        timestep = torch.randint(1, self.max_timesteps, (x.shape[0],), device=x.device)
        # Add noise to the input data at the chosen timesteps
        x_t = self.apply_noise(x, timestep)
        # Denoise the noised data by sampling from the model (without adding noise)
        _, x_0_pred = self.compute_noise_and_x_0_pred(x_t, condition, timestep)
        # Compute mean squared error loss between the denoised prediction and the original data
        loss = torch.nn.functional.mse_loss(x_0_pred, x)
        return loss
    
    def state_dict(self, *args, **kwargs):
        if self.model is not None and len(self.model) > 0:
            model = self.model[0]
            self.set_model(None)
            state_dict = super().state_dict(*args, **kwargs)
            self.set_model(model)
            return state_dict
        else:
            return super().state_dict(*args, **kwargs)