from types import SimpleNamespace

import pytest
import torch

peft = pytest.importorskip("peft")

from models.b2b_model import B2BModel
from models.modules.b2b_generator import B2BGenerator
from models.modules.vit.vit import JiT


def _make_opt(tmp_path, strict=False, g_netg="vit"):
    return SimpleNamespace(
        alg_b2b_lora=True,
        alg_b2b_lora_rank=2,
        alg_b2b_lora_alpha=4,
        alg_b2b_lora_dropout=0.0,
        alg_b2b_lora_target_modules=["attn.qkv", "attn.proj", "mlp.w12", "mlp.w3"],
        alg_b2b_denoise_timesteps=[1],
        alg_b2b_P_std=0.8,
        alg_b2b_t_eps=5e-2,
        alg_b2b_use_gt_prob=0.1,
        alg_b2b_multi_dataset_class_conditioning=False,
        alg_b2b_P_mean=-0.8,
        alg_b2b_noise_scale=-1.0,
        alg_b2b_cfg_scale=1.0,
        alg_b2b_clip_denoised=False,
        alg_b2b_disable_inference_clipping=False,
        alg_diffusion_dropout_prob=0.0,
        isTrain=True,
        G_netG=g_netg,
        model_load_no_strictness=strict,
        train_G_ema=False,
        G_vit_num_classes=1,
        checkpoints_dir=str(tmp_path),
        name="b2b_lora",
    )


def _make_generator(opt):
    jit = JiT(
        input_size=16,
        patch_size=8,
        in_channels=3,
        hidden_size=16,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        num_classes=1,
        bottleneck_dim=8,
        in_context_len=0,
    )
    return B2BGenerator(jit, "", image_size=16, G_ngf=16, opt=opt)


def _make_lora_model(tmp_path, strict=False):
    opt = _make_opt(tmp_path, strict=strict)
    model = B2BModel.__new__(B2BModel)
    model.opt = opt
    model.device = torch.device("cpu")
    model.gpu_ids = []
    model.use_cuda = False
    model.isTrain = True
    model.save_dir = str(tmp_path)
    model.model_names = ["G_A"]
    model.netG_A = _make_generator(opt)
    model._apply_b2b_lora()
    return model


def test_b2b_lora_wraps_only_lora_parameters_for_training(tmp_path):
    model = _make_lora_model(tmp_path)

    trainable = [
        name for name, param in model.netG_A.named_parameters() if param.requires_grad
    ]
    frozen = [
        name
        for name, param in model.netG_A.named_parameters()
        if not param.requires_grad
    ]

    assert trainable
    assert frozen
    assert all("lora_" in name for name in trainable)
    assert len(model._iter_trainable_generator_parameters()) == len(trainable)


def test_b2b_lora_saves_merged_full_checkpoint(tmp_path):
    model = _make_lora_model(tmp_path)

    model.save_networks("latest")

    state_dict = torch.load(tmp_path / "latest_net_G_A.pth", map_location="cpu")
    assert "b2b_model.blocks.0.attn.qkv.weight" in state_dict
    assert not any("lora_" in key for key in state_dict)
    assert not any("base_model.model" in key for key in state_dict)
    assert not any("base_layer" in key for key in state_dict)
    assert any("lora_" in key for key in model.netG_A.state_dict())


def test_b2b_lora_loads_raw_full_checkpoint_into_wrapped_model(tmp_path):
    opt = _make_opt(tmp_path)
    raw_net = _make_generator(opt)
    with torch.no_grad():
        raw_net.b2b_model.blocks[0].attn.qkv.weight.fill_(0.125)
        raw_net.b2b_model.blocks[0].attn.qkv.bias.fill_(0.25)
    torch.save(raw_net.state_dict(), tmp_path / "latest_net_G_A.pth")

    model = _make_lora_model(tmp_path)
    model.load_networks("latest", load_dir=str(tmp_path))

    qkv = model.netG_A.b2b_model.base_model.model.blocks[0].attn.qkv.base_layer
    assert torch.allclose(qkv.weight, raw_net.b2b_model.blocks[0].attn.qkv.weight)
    assert torch.allclose(qkv.bias, raw_net.b2b_model.blocks[0].attn.qkv.bias)


def test_b2b_lora_rejects_non_jit_backbone(tmp_path):
    opt = _make_opt(tmp_path, g_netg="unet_mha")

    with pytest.raises(ValueError, match="only supported"):
        B2BModel.after_parse(opt)
