from rsl_rl.modules.actor_critic import ActorCritic as BaseActorCritic
import torch
from torch.distributions import Normal

class ActorCritic(BaseActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self.distribution = Normal(mean, std + 1e-6) # add small epsilon to avoid log(0)
