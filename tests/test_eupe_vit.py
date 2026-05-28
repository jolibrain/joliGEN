from types import SimpleNamespace

import pytest
import torch

from models import diffusion_networks
from models.modules.vit.eupe_vit import EUPEDiffusionViT, EUPEDiffusionViTVideo


def _make_define_g_opt(**overrides):
    values = {
        "G_vit_variant": "EUPE-T/16",
        "G_vit_num_classes": 2,
        "G_vit_pretrained_weights": "",
        "G_vit_depth": 1,
        "train_continue": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_eupe_diffusion_vit_forward_shape():
    model = EUPEDiffusionViT(
        input_size=32,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        embed_dim=192,
        depth=1,
        num_heads=3,
        num_classes=2,
    )
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    y = torch.tensor([0, 1])

    out = model(x, t, y)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_eupe_diffusion_vit_video_forward_shape():
    model = EUPEDiffusionViTVideo(
        input_size=32,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        embed_dim=192,
        depth=1,
        num_heads=3,
        num_classes=2,
        max_frames=3,
        motion_num_heads=3,
        motion_num_layers=1,
    )
    x = torch.randn(1, 2, 3, 32, 32)
    t = torch.rand(1)
    y = torch.tensor([1])

    out = model(x, t, y)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    ("variant", "hidden_size"),
    [("EUPE-T/16", 192), ("EUPE-S/16", 384), ("EUPE-B/16", 768)],
)
def test_define_g_accepts_eupe_variants(variant, hidden_size):
    opt = _make_define_g_opt(G_vit_variant=variant)

    net = diffusion_networks.define_G(
        model_type="b2b",
        model_input_nc=3,
        model_output_nc=3,
        G_netG="vit",
        G_nblocks=0,
        data_crop_size=32,
        G_norm="batch",
        G_diff_n_timestep_train=1,
        G_diff_n_timestep_test=1,
        G_dropout=0,
        G_ngf=64,
        G_unet_mha_num_heads=1,
        G_unet_mha_num_head_channels=1,
        G_unet_mha_res_blocks=1,
        G_unet_mha_channel_mults=[1],
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
        alg_diffusion_task="inpainting",
        alg_diffusion_cond_embed="y_t",
        alg_diffusion_cond_embed_dim=256,
        alg_diffusion_ref_embed_net="",
        alg_diffusion_ddpm_cm_ft=False,
        model_prior_321_backwardcompatibility=False,
        f_s_semantic_nclasses=-1,
        opt=opt,
    )

    assert net.b2b_model.hidden_size == hidden_size


def test_eupe_pretrained_loader_adapts_extra_mask_channel(tmp_path):
    source = EUPEDiffusionViT(
        input_size=32,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        embed_dim=192,
        depth=1,
        num_heads=3,
        num_classes=2,
    )
    ckpt = tmp_path / "eupe_source.pth"
    torch.save({"module.teacher." + k: v for k, v in source.state_dict().items()}, ckpt)

    target = EUPEDiffusionViT(
        input_size=32,
        patch_size=16,
        in_channels=4,
        out_channels=3,
        embed_dim=192,
        depth=1,
        num_heads=3,
        num_classes=2,
        pretrained_weights=str(ckpt),
    )

    assert torch.allclose(
        target.patch_embed.proj.weight[:, :3], source.patch_embed.proj.weight
    )
    assert torch.count_nonzero(target.patch_embed.proj.weight[:, 3:]).item() == 0


def test_eupe_pretrained_loader_adapts_rgb_condition_and_target(tmp_path):
    source = EUPEDiffusionViT(
        input_size=32,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        embed_dim=192,
        depth=1,
        num_heads=3,
        num_classes=2,
    )
    ckpt = tmp_path / "eupe_source.pth"
    torch.save(source.state_dict(), ckpt)

    target = EUPEDiffusionViT(
        input_size=32,
        patch_size=16,
        in_channels=6,
        out_channels=3,
        embed_dim=192,
        depth=1,
        num_heads=3,
        num_classes=2,
        pretrained_weights=str(ckpt),
    )

    expected = source.patch_embed.proj.weight / 2
    assert torch.allclose(target.patch_embed.proj.weight[:, :3], expected)
    assert torch.allclose(target.patch_embed.proj.weight[:, 3:6], expected)
