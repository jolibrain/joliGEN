from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models import diffusion_networks
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


def test_jit_vid_uses_last_layer_motion_by_default():
    model = JiTViD(**_jit_kwargs(num_register_tokens=0), out_channels=3, max_frames=2)

    assert model.motion_every == 0
    assert model.motion_insert_layers == [1]
    assert hasattr(model, "motion_module")
    assert not hasattr(model, "motion_modules")


def test_jit_vid_motion_every_one_adds_module_after_each_block():
    model = JiTViD(
        **_jit_kwargs(num_register_tokens=0),
        out_channels=3,
        max_frames=2,
        motion_every=1,
    )

    assert model.motion_every == 1
    assert model.motion_insert_layers == [0, 1]
    assert len(model.motion_modules) == 2


def test_jit_vid_motion_every_two_includes_final_block():
    kwargs = _jit_kwargs(num_register_tokens=0)
    kwargs["depth"] = 3
    model = JiTViD(**kwargs, out_channels=3, max_frames=2, motion_every=2)

    assert model.motion_every == 2
    assert model.motion_insert_layers == [1, 2]
    assert len(model.motion_modules) == 2


def test_jit_vid_rejects_negative_motion_every():
    with pytest.raises(ValueError, match="motion_every must be >= 0"):
        JiTViD(
            **_jit_kwargs(num_register_tokens=0),
            out_channels=3,
            max_frames=2,
            motion_every=-1,
        )


def test_define_g_passes_vit_vid_motion_every(monkeypatch):
    captured = {}

    class FakeJiTViD(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured.update(kwargs)
            self.out_channel = kwargs["out_channels"]
            self.num_classes = kwargs["num_classes"]

    monkeypatch.setattr(diffusion_networks, "JiTViD", FakeJiTViD)

    opt = SimpleNamespace(
        G_vit_variant="",
        G_vit_vid_motion_every=2,
        G_vit_num_classes=3,
        alg_b2b_P_mean=-0.8,
        alg_b2b_P_std=0.8,
        alg_b2b_noise_scale=-1.0,
        alg_b2b_t_eps=5e-2,
        alg_b2b_cfg_scale=1.0,
        alg_b2b_clip_denoised=False,
        alg_b2b_disable_inference_clipping=False,
    )

    net = diffusion_networks.define_G(
        model_type="b2b",
        model_input_nc=3,
        model_output_nc=3,
        G_netG="vit_vid",
        G_nblocks=0,
        data_crop_size=16,
        G_norm="batch",
        G_diff_n_timestep_train=1,
        G_diff_n_timestep_test=1,
        G_dropout=0.0,
        G_ngf=16,
        G_unet_mha_num_heads=1,
        G_unet_mha_num_head_channels=-1,
        G_unet_mha_res_blocks=1,
        G_unet_mha_channel_mults=[],
        G_unet_mha_attn_res=[],
        G_hdit_depths=[],
        G_hdit_widths=[],
        G_hdit_patch_size=4,
        G_attn_nb_mask_attn=0,
        G_attn_nb_mask_input=0,
        G_spectral=False,
        G_unet_vid_max_sequence_length=2,
        G_unet_vid_num_attention_heads=1,
        G_unet_vid_num_transformer_blocks=1,
        jg_dir="",
        G_padding_type="reflect",
        G_config_segformer="",
        G_unet_mha_norm_layer="groupnorm",
        G_unet_mha_group_norm_size=32,
        G_uvit_num_transformer_blocks=1,
        G_unet_mha_vit_efficient=False,
        alg_palette_sampling_method="",
        alg_diffusion_task="",
        alg_diffusion_cond_embed="",
        alg_diffusion_cond_embed_dim=32,
        alg_diffusion_ref_embed_net="",
        alg_diffusion_ddpm_cm_ft=False,
        model_prior_321_backwardcompatibility=False,
        opt=opt,
    )

    assert captured["motion_every"] == 2
    assert net.b2b_model.out_channel == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="JiT attention requires CUDA")
def test_jit_register_tokens_keep_output_shape_on_forward():
    model = JiT(**_jit_kwargs(num_register_tokens=3)).cuda()
    x = torch.randn(2, 3, 16, 16, device="cuda")
    t = torch.rand(2, device="cuda")
    y = torch.zeros(2, dtype=torch.long, device="cuda")

    output = model(x, t, y)

    assert output.shape == x.shape


@pytest.mark.parametrize("motion_every", [0, 1, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="JiT attention requires CUDA")
def test_jit_vid_register_tokens_keep_output_shape_on_forward(motion_every):
    model = JiTViD(
        **_jit_kwargs(num_register_tokens=3),
        out_channels=3,
        max_frames=2,
        motion_every=motion_every,
    ).cuda()
    x = torch.randn(2, 2, 3, 16, 16, device="cuda")
    t = torch.rand(2, device="cuda")
    y = torch.zeros(2, dtype=torch.long, device="cuda")

    output = model(x, t, y)

    assert output.shape == x.shape
