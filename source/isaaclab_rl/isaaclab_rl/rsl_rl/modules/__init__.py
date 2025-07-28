from rsl_rl.modules import (
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacherRecurrent,
)
from .actor_critic import ActorCritic
from .twin_delayed import TwinDelayed
from .actor_critic_ou import ActorCriticOU
from .actor_double_critic import ActorDoubleCritic
from .actor_critic_moe import ActorCriticMoE
from .actor_critic_mop import ActorCriticMoP
from .actor_critic_pnn import ActorCriticPNN
from .actor_critic_dp import ActorCriticDP
from .student_teacher import StudentTeacher

import torch
class _ModuleWrapper(torch.nn.Module):
    def __init__(self, module, normalizer=None):
        super().__init__()
        self.module = module
        self.normalizer = normalizer

    def forward(self, x):
        if self.normalizer is not None:
            x = self.normalizer(x)
        return self.module(x)

def resolve_module(checkpoint_path) -> tuple[torch.nn.Module, tuple, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_cfg = checkpoint["policy_cfg"]
    policy_class = eval(policy_cfg.pop("class_name"))
    policy_args = policy_cfg.pop("_args")
    policy = policy_class(*policy_args, **policy_cfg)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()

    if hasattr(policy, "student"):
        policy = policy.student
    if hasattr(policy, "actor"):
        policy = policy.actor
    if hasattr(policy, "act_inference"):
        policy = policy.act_inference
    if 'obs_norm_state_dict' in checkpoint:
        emp_state_dict = checkpoint['obs_norm_state_dict']
        num_obs = policy_args[0] if isinstance(policy_args[0], int) else policy_args[0][0]
        
        emperical_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8)
        emperical_normalizer.load_state_dict(emp_state_dict)
        policy = _ModuleWrapper(policy, emperical_normalizer).eval()

    return policy, policy_args, policy_cfg

__all__ = ["ActorCritic", "ActorCriticRecurrent", "EmpiricalNormalization", "StudentTeacher", "StudentTeacherRecurrent", "TwinDelayed", "ActorCriticOU", "ActorDoubleCritic", "ActorCriticMoE", "ActorCriticMoP", "ActorCriticPNN", "ActorCriticDP"]
