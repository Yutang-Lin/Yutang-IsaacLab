from rsl_rl.algorithms import Distillation
from .hrpo import HRPO  
from .ppo import PPO
from .ppo_kl import PPOKL
from .amp import AmpReward
from .td3 import TD3
from .double_ppo import DoublePPO

__all__ = ["PPO", "PPOKL", "Distillation", "HRPO", "AmpReward", "TD3", "DoublePPO"]
