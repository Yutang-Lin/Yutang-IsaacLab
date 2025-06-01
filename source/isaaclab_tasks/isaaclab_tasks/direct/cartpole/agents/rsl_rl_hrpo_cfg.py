# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "BaseRunner"

    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="HRPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=1.0,
        entropy_coef=0.0,
        kl_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma_f=0.99,
        gamma_r=1.0,
        alpha=0.8,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
