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
    Uses SUM + manual division so it's compatible with both NCCL and Gloo backends
    (Gloo does not support ReduceOp.AVG).
    """
    grads = [param.grad.view(-1) for param in network.parameters() if param.grad is not None]
    if len(grads) == 0:
        return
    all_grads = torch.cat(grads)

    # Sum gradients across all ranks then divide by world_size for the average
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads.div_(torch.distributed.get_world_size())

    # Write averaged gradients back into each parameter's grad buffer
    offset = 0
    for param in network.parameters():
        if param.grad is not None:
            numel = param.numel()
            param.grad.data.copy_(all_grads[offset: offset + numel].view_as(param.grad.data))
            offset += numel


def reduce_gradients_async(network):
    """Non-blocking variant of :func:`reduce_gradients`.

    Fires a single ``all_reduce(async_op=True)`` over the concatenated
    gradients of ``network`` and returns an opaque handle tuple. Call
    :func:`finish_async_reduce` on the handle before reading / clipping /
    stepping the gradients — that's when the collective actually
    synchronises and the averaged grads get scattered back.

    Pattern (pipelines reduce with the next block's compute):

        h = reduce_gradients_async(net_A)
        # ... other work (including net_B's backward) ...
        finish_async_reduce(h)
        optA.step()

    Returns None when there are no grads to reduce OR when distributed
    is not initialised (caller can treat ``if h is None: optA.step()``).
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    params_with_grad = [p for p in network.parameters() if p.grad is not None]
    if not params_with_grad:
        return None
    flat = torch.cat([p.grad.view(-1) for p in params_with_grad])
    handle = torch.distributed.all_reduce(
        flat, op=torch.distributed.ReduceOp.SUM, async_op=True
    )
    return (handle, flat, params_with_grad)


def finish_async_reduce(handle_tuple):
    """Wait on an async all_reduce handle and scatter the averaged grads back."""
    if handle_tuple is None:
        return
    handle, flat, params_with_grad = handle_tuple
    handle.wait()
    flat.div_(torch.distributed.get_world_size())
    offset = 0
    for p in params_with_grad:
        n = p.grad.numel()
        p.grad.data.copy_(flat[offset: offset + n].view_as(p.grad.data))
        offset += n


def reduce_gradients_merged_async(networks):
    """Single async all_reduce across the grads of multiple networks.

    Fires one ``all_reduce(async_op=True)`` over the concatenated grads of
    every network in ``networks`` (accepts both DDP-wrapped and plain
    ``nn.Module``). Pairs with :func:`finish_merged_async_reduce` which
    scatters averaged grads back into each parameter in the same order.

    This collapses 4-6 per-network allreduces into a single collective,
    removing 3-5× the per-call NCCL latency each update. Intended for use
    with ``DistributedDataParallel.no_sync()`` contexts — callers disable
    DDP's bucketed in-backward reduce so this function does the only sync.

    Returns None when distributed is not initialised or no grads exist;
    callers should fall back to plain ``opt.step()`` in that case.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    params_with_grad = []
    for net in networks:
        for p in net.parameters():
            if p.grad is not None:
                params_with_grad.append(p)
    if not params_with_grad:
        return None
    flat = torch.cat([p.grad.view(-1) for p in params_with_grad])
    handle = torch.distributed.all_reduce(
        flat, op=torch.distributed.ReduceOp.SUM, async_op=True
    )
    return (handle, flat, params_with_grad)


def finish_merged_async_reduce(handle_tuple):
    """Wait on a merged async all_reduce and scatter averaged grads back."""
    finish_async_reduce(handle_tuple)


def zero_grads_if_nonfinite(loss, *networks) -> bool:
    """Zero the gradients of ``networks`` locally if ``loss`` is not finite.

    Returns True when the zeroing happened. Designed for DDP + collective
    gradient averaging: a rank whose backward produced NaN/Inf grads zeroes
    its local contribution **but still calls ``reduce_gradients`` afterward**,
    so the collective ``all_reduce`` doesn't deadlock. Averaged grad = sum
    of finite ranks' grads + zeros from NaN ranks, then divided by
    world_size — slightly down-weighted but never poisoned.

    The alternative (global skip when anyone has NaN) discards all ranks'
    valid work that step; this variant preserves it at the cost of a
    slightly smaller effective batch on NaN iters.
    """
    import torch as _torch
    if _torch.isfinite(loss).all():
        return False
    for net in networks:
        for p in net.parameters():
            if p.grad is not None:
                p.grad.zero_()
    return True