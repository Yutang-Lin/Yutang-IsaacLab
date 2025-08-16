from .empirical_normalization import EmpiricalNormalization
from .actor_critic import ActorCritic
from .actor_critic_recurrent import ActorCriticRecurrent
from .twin_delayed import TwinDelayed
from .actor_critic_ou import ActorCriticOU
from .actor_double_critic import ActorDoubleCritic
from .actor_critic_moe import ActorCriticMoE
from .actor_critic_mop import ActorCriticMoP
from .actor_critic_pnn import ActorCriticPNN
from .actor_critic_dp import ActorCriticDP
from .actor_critic_transformer import ActorCriticTransformer
from .actor_critic_transformer_latent import ActorCriticTransformerLatent
from .actor_critic_tf_recurrent import ActorCriticTFRecurrent
from .actor_critic_tf_recurrent_ml import ActorCriticTFRecurrentML
from .actor_critic_tf_recurrent_ll import ActorCriticTFRecurrentLL
from .actor_critic_tf_recurrent_latent import ActorCriticTFRecurrentLatent
from .actor_critic_dp_transformer import ActorCriticDPTransformer
from .student_teacher import StudentTeacher
from .student_teacher_recurrent import StudentTeacherRecurrent

import torch
from typing import Callable

class _ModuleWrapper(torch.nn.Module):
    def __init__(self, module_ori, module, normalizer=None):
        super().__init__()
        self.module_ori: ActorCritic = module_ori
        self.module = module
        self.normalizer = normalizer

    def forward(self, x):
        if self.normalizer is not None:
            x = self.normalizer(x)
        return self.module(x)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except:
            return getattr(self.module_ori, name)

def resolve_module(checkpoint_path, device="cpu") -> tuple[_ModuleWrapper, tuple, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_cfg = checkpoint["policy_cfg"]
    policy_class = eval(policy_cfg.pop("class_name"))
    policy_args = policy_cfg.pop("_args")
    policy = policy_class(*policy_args, **policy_cfg)
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.to(device)
    policy.eval()
    module_ori = policy

    if hasattr(policy, "student"):
        policy = policy.student
    if hasattr(policy, "act_inference"):
        policy = policy.act_inference
    elif hasattr(policy, "actor"):
        policy = policy.actor
    else:
        raise ValueError("Policy has no act_inference, actor, or student attribute")

    if 'obs_norm_state_dict' in checkpoint:
        emp_state_dict = checkpoint['obs_norm_state_dict']
        num_obs = policy_args[0] if isinstance(policy_args[0], int) else policy_args[0][0]
        
        emperical_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8)
        emperical_normalizer.load_state_dict(emp_state_dict)
    else:
        emperical_normalizer = None
    policy = _ModuleWrapper(module_ori, policy, emperical_normalizer).eval()

    return policy, policy_args, policy_cfg

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "ActorCriticOU",
    "ActorCriticDP",
    "ActorCriticDPTransformer",
    "ActorCriticTFRecurrent",
    "ActorCriticTFRecurrentML",
    "ActorCriticTFRecurrentLL",
    "ActorCriticTFRecurrentLatent",
    "ActorCriticTransformerLatent",
    "ActorCriticMoE",
    "ActorCriticMoP",
    "ActorCriticPNN",
    "EmpiricalNormalization",
    "StudentTeacher",
    "StudentTeacherRecurrent",
    "TwinDelayed",
    "ActorDoubleCritic",
    "ActorCriticTransformer",
]
