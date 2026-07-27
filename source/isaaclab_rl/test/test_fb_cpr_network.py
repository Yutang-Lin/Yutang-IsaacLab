import os
os.environ["ENABLE_ISAACLAB"] = "False"

import importlib.util
import sys
from pathlib import Path

import gymnasium
import pytest
import torch

from isaaclab_rl.rsl_rl.fb_cpr.fb_networks import test_fb_networks

_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaaclab_rl/rsl_rl/modules/fb_cpr_policy.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_fb_cpr_policy_module", _POLICY_PATH
)
_POLICY_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _POLICY_MODULE
_SPEC.loader.exec_module(_POLICY_MODULE)
FourierMLP = _POLICY_MODULE.FourierMLP
ForwardMap = _POLICY_MODULE.ForwardMap
ScalarMLP = _POLICY_MODULE.ScalarMLP


def _gamma_forward_map(embed_type: str) -> ForwardMap:
    return ForwardMap(
        gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(7,)),
        z_dim=5,
        action_dim=3,
        hidden_dim=16,
        model="simple",
        hidden_layers=1,
        embedding_layers=2,
        num_parallel=2,
        gamma_embed_dim=8,
        gamma_embed_type=embed_type,
    )


@pytest.mark.parametrize(
    ("embed_type", "embed_class", "first_input_dim"),
    [("fourier", FourierMLP, 32), ("mlp", ScalarMLP, 1)],
)
def test_forward_map_gamma_embedding_types(
    embed_type, embed_class, first_input_dim
):
    module = _gamma_forward_map(embed_type)

    assert isinstance(module.embed_gamma, embed_class)
    assert module.embed_gamma.mlp[0].in_features == first_input_dim
    output = module(
        torch.randn(4, 7),
        torch.randn(4, 5),
        torch.randn(4, 3),
        torch.linspace(0.6, 0.99, 4),
    )
    assert output.shape == (2, 4, 5)
    assert torch.isfinite(output).all()


def test_forward_map_rejects_unknown_gamma_embedding_type():
    with pytest.raises(ValueError, match="gamma_embed_type"):
        _gamma_forward_map("unknown")

if __name__ == "__main__":
    test_fb_networks()
