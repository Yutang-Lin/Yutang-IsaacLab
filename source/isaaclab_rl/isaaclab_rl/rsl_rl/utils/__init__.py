import torch

class TensorDict(dict):
    def unsqueeze(self, *args, **kwargs):
        return TensorDict({k: v.unsqueeze(*args, **kwargs) if v is not None else None for k, v in self.items()})
    
    def transpose(self, *args, **kwargs):
        return TensorDict({k: v.transpose(*args, **kwargs) if v is not None else None for k, v in self.items()})
    
    def contiguous(self, *args, **kwargs):
        return TensorDict({k: v.contiguous(*args, **kwargs) if v is not None else None for k, v in self.items()})

    def shape(self, item):
        return self[item].shape

def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    if act_name == "elu":
        return torch.nn.ELU()
    elif act_name == "selu":
        return torch.nn.SELU()
    elif act_name == "relu":
        return torch.nn.ReLU()
    elif act_name == "crelu":
        return torch.nn.CELU()
    elif act_name == "lrelu":
        return torch.nn.LeakyReLU()
    elif act_name == "tanh":
        return torch.nn.Tanh()
    elif act_name == "gelu":
        return torch.nn.GELU(approximate="tanh")
    elif act_name == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_name == "identity":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Invalid activation function '{act_name}'.")