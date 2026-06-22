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

def _ensure_grads(params):
    """Materialize a zero grad for any param whose ``.grad`` is None.

    CRITICAL for collective gradient sync: the all_reduce buffer is built from
    the per-parameter grads, so its SIZE must be identical on every rank. If a
    param receives a grad on one rank but not another (e.g. masked / data-
    dependent paths in the transformer actor, where some positions or the z
    token contribute no gradient on a given rank's batch), filtering on
    ``grad is not None`` yields different-sized buffers and the collective
    DEADLOCKS. Zero-filling makes every rank iterate the SAME fixed parameter
    set, so the buffer size and order are rank-consistent.

    Reuses a PERSISTENT per-param zero buffer (``_zero_grad_buf``) instead of
    allocating a fresh ``zeros_like`` each step. Allocating a new grad tensor
    every iteration (params are zeroed with ``set_to_none=True``) changes the
    grad tensor identity/address each step, which churns ``torch.compile`` /
    CUDA-graph guards and triggers recompiles — exploding learn time. The
    persistent buffer keeps a stable address; we only zero it in place when it
    is actually reused (a param whose grad genuinely flowed keeps its own grad).
    """
    for p in params:
        if p.grad is None:
            buf = getattr(p, "_zero_grad_buf", None)
            if buf is None or buf.shape != p.shape or buf.device != p.device:
                buf = torch.zeros_like(p)
                p._zero_grad_buf = buf
            else:
                buf.zero_()
            p.grad = buf


def reduce_gradients(network):
    """Collect gradients from all GPUs and average them.

    This function is called after the backward pass to synchronize the gradients across all GPUs.
    Uses SUM + manual division so it's compatible with both NCCL and Gloo backends
    (Gloo does not support ReduceOp.AVG).
    """
    params = list(network.parameters())
    if len(params) == 0:
        return
    # Zero-fill missing grads so the buffer is rank-consistent (see _ensure_grads).
    _ensure_grads(params)
    all_grads = torch.cat([p.grad.view(-1) for p in params])

    # Sum gradients across all ranks then divide by world_size for the average
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads.div_(torch.distributed.get_world_size())

    # Write averaged gradients back into each parameter's grad buffer
    offset = 0
    for param in params:
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
    params = list(network.parameters())
    if not params:
        return None
    # Zero-fill missing grads so the all_reduce buffer is the SAME size/order on
    # every rank — otherwise data-dependent grad-None sets deadlock the
    # collective (see _ensure_grads). This is the path the transformer actor
    # uses (its forward_window bypasses DDP's hooks), where masked positions /
    # the z token can leave a param grad-None on some ranks but not others.
    _ensure_grads(params)
    flat = torch.cat([p.grad.view(-1) for p in params])
    handle = torch.distributed.all_reduce(
        flat, op=torch.distributed.ReduceOp.SUM, async_op=True
    )
    return (handle, flat, params)


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
    params = []
    for net in networks:
        params.extend(net.parameters())
    if not params:
        return None
    # Zero-fill missing grads so the merged buffer is rank-consistent in size
    # and order (see _ensure_grads) — a grad-None param on only some ranks would
    # otherwise deadlock the single fused collective.
    _ensure_grads(params)
    flat = torch.cat([p.grad.view(-1) for p in params])
    handle = torch.distributed.all_reduce(
        flat, op=torch.distributed.ReduceOp.SUM, async_op=True
    )
    return (handle, flat, params)


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