import pytest
import torch

from models.modules.vit.vit import JiT
from models.modules.vit.vit_vid import JiTViD


def _jit_kwargs(num_register_tokens):
    return {
        "input_size": 16,
        "patch_size": 8,
        "in_channels": 3,
        "hidden_size": 32,
        "depth": 2,
        "num_heads": 2,
        "bottleneck_dim": 32,
        "in_context_len": 2,
        "in_context_start": 1,
        "num_classes": 2,
        "num_register_tokens": num_register_tokens,
    }


def test_jit_register_tokens_are_disabled_by_default():
    model = JiT(**_jit_kwargs(num_register_tokens=0))

    assert model.num_register_tokens == 0
    assert not hasattr(model, "register_tokens")


def test_jit_register_tokens_are_configurable():
    model = JiT(**_jit_kwargs(num_register_tokens=3))

    assert model.num_register_tokens == 3
    assert model.register_tokens.shape == (1, 3, 32)


def test_jit_vid_register_tokens_are_configurable():
    model = JiTViD(**_jit_kwargs(num_register_tokens=4), out_channels=3, max_frames=2)

    assert model.num_register_tokens == 4
    assert model.register_tokens.shape == (1, 4, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="JiT attention requires CUDA")
def test_jit_register_tokens_keep_output_shape_on_forward():
    model = JiT(**_jit_kwargs(num_register_tokens=3)).cuda()
    x = torch.randn(2, 3, 16, 16, device="cuda")
    t = torch.rand(2, device="cuda")
    y = torch.zeros(2, dtype=torch.long, device="cuda")

    output = model(x, t, y)

    assert output.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="JiT attention requires CUDA")
def test_jit_vid_register_tokens_keep_output_shape_on_forward():
    model = JiTViD(
        **_jit_kwargs(num_register_tokens=3), out_channels=3, max_frames=2
    ).cuda()
    x = torch.randn(2, 2, 3, 16, 16, device="cuda")
    t = torch.rand(2, device="cuda")
    y = torch.zeros(2, dtype=torch.long, device="cuda")

    output = model(x, t, y)

    assert output.shape == x.shape
