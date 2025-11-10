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
    
def broadcast_parameters(policy):
    """Broadcast model parameters to all GPUs."""
    # obtain the model parameters on current GPU
    model_params = [policy.state_dict()]
    # broadcast the model parameters
    torch.distributed.broadcast_object_list(model_params, src=0)
    # load the model parameters on all GPUs from source GPU
    policy.load_state_dict(model_params[0])

def reduce_gradients(network):
    """Collect gradients from all GPUs and average them.

    This function is called after the backward pass to synchronize the gradients across all GPUs.
    """
    # Create a tensor to store the gradients
    grads = [param.grad.view(-1) for param in network.parameters() if param.grad is not None]
    all_grads = torch.cat(grads)

    # Average the gradients across all GPUs
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.AVG)

    # Get all parameters
    all_params = network.parameters()

    # Update the gradients for all parameters with the reduced gradients
    offset = 0
    for param in all_params:
        if param.grad is not None:
            numel = param.numel()
            # copy data back from shared buffer
            param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
            # update the offset for the next parameter
            offset += numel