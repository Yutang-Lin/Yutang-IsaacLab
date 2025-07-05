from rsl_rl.algorithms.ppo import PPO as RslRlPPO

class PPO(RslRlPPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
            num_learning_epochs=kwargs.pop("num_learning_epochs", 1),
            num_mini_batches=kwargs.pop("num_mini_batches", 1),
            clip_param=kwargs.pop("clip_param", 0.2),
            gamma=kwargs.pop("gamma", 0.998),
            lam=kwargs.pop("lam", 0.95),
            value_loss_coef=kwargs.pop("value_loss_coef", 1.0),
            entropy_coef=kwargs.pop("entropy_coef", 0.0),
            learning_rate=kwargs.pop("learning_rate", 1e-3),
            max_grad_norm=kwargs.pop("max_grad_norm", 1.0),
            use_clipped_value_loss=kwargs.pop("use_clipped_value_loss", True),
            schedule=kwargs.pop("schedule", "fixed"),
            desired_kl=kwargs.pop("desired_kl", 0.01),
            device=kwargs.pop("device", "cpu"),
            normalize_advantage_per_mini_batch=kwargs.pop("normalize_advantage_per_mini_batch", False),
            # RND parameters
            rnd_cfg=kwargs.pop("rnd_cfg", None),
            # Symmetry parameters
            symmetry_cfg=kwargs.pop("symmetry_cfg", None),
            # Distributed training parameters
            multi_gpu_cfg=kwargs.pop("multi_gpu_cfg", None),
        )

    def compute_returns(self, last_critic_obs, **kwargs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )