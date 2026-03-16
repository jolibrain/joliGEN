import sys

import pytest
import torch
import torch.nn as nn

sys.path.append(sys.path[0] + "/..")

from models.mat_model import MATModel
from models.modules.mat import Discriminator, Generator
from options.train_options import TrainOptions
import models.mat_model as mat_model_module


def make_mat_opt(tmp_path, **overrides):
    config = {
        "name": "mat_utest",
        "dataroot": str(tmp_path),
        "checkpoints_dir": str(tmp_path),
        "gpu_ids": "-1",
        "model_type": "mat",
        "data_dataset_mode": "self_supervised_labeled_mask_online",
        "data_crop_size": 256,
        "data_load_size": 256,
        "data_max_dataset_size": 1,
        "train_batch_size": 1,
        "test_batch_size": 1,
        "output_display_env": "mat_utest",
        "output_display_id": 0,
        "train_n_epochs": 1,
        "train_n_epochs_decay": 0,
        "data_num_threads": 0,
    }
    config.update(overrides)
    opt = TrainOptions().parse_json(config, save_config=False)
    opt.optim = lambda opt, params, lr, betas, weight_decay, eps: torch.optim.Adam(
        params,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
    )
    opt.use_cuda = False
    return opt


def make_batch(path="/tmp/sample.png", b_mask_value=1):
    real_a = torch.randn(1, 3, 256, 256)
    real_b = torch.randn(1, 3, 256, 256)
    a_mask = torch.zeros(1, 1, 256, 256)
    b_mask = torch.zeros(1, 1, 256, 256)
    b_mask[:, :, 64:128, 96:160] = b_mask_value

    return {
        "A": real_a,
        "B": real_b,
        "A_img_paths": [path],
        "A_label_mask": a_mask,
        "B_label_mask": b_mask,
    }


def test_mat_parse_defaults(tmp_path):
    opt = make_mat_opt(tmp_path)

    assert opt.model_type == "mat"
    assert opt.D_netDs == ["none"]
    assert opt.train_G_lr == pytest.approx(0.001)
    assert opt.train_D_lr == pytest.approx(0.001)
    assert opt.train_beta1 == pytest.approx(0.0)
    assert opt.train_beta2 == pytest.approx(0.99)
    assert opt.train_G_ema is True


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"data_dataset_mode": "unaligned"}, "inpainting datasets"),
        ({"train_export_jit": True}, "does not support joliGEN generator export"),
        ({"model_input_nc": 1}, "RGB inputs and outputs only"),
    ],
)
def test_mat_parse_rejects_unsupported_combinations(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        make_mat_opt(tmp_path, **overrides)


def test_vendored_mat_generator_and_discriminator_shapes():
    image = torch.randn(1, 3, 256, 256)
    mask_keep = (torch.rand(1, 1, 256, 256) > 0.5).float()
    z = torch.randn(1, 512)

    net_g = Generator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=256,
        img_channels=3,
    )
    net_d = Discriminator(
        c_dim=0,
        img_resolution=256,
        img_channels=3,
        mbstd_group_size=1,
    )

    fake_b, fake_b_stg1 = net_g(image, mask_keep, z, None, return_stg1=True)
    logits, logits_stg1 = net_d(fake_b, mask_keep, fake_b_stg1, None)

    assert fake_b.shape == image.shape
    assert fake_b_stg1.shape == image.shape
    assert logits.shape == (1, 1)
    assert logits_stg1.shape == (1, 1)
    assert torch.isfinite(fake_b).all()
    assert torch.isfinite(logits).all()
    assert fake_b.max().item() <= 1.0
    assert fake_b.min().item() >= -1.0
    assert fake_b_stg1.max().item() <= 1.0
    assert fake_b_stg1.min().item() >= -1.0


def test_vendored_mat_generator_mask_class_conditioning_keeps_rgb_output():
    image = torch.randn(1, 3, 256, 256)
    mask_keep = (torch.rand(1, 1, 256, 256) > 0.5).float()
    mask_class = torch.zeros(1, 1, 256, 256)
    mask_class[:, :, 64:128, 96:160] = 2.0
    z = torch.randn(1, 512)

    net_g = Generator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=256,
        img_channels=3,
        synthesis_kwargs={"mask_class_channels": 1},
    )

    fake_b, fake_b_stg1 = net_g(
        image, mask_keep, z, None, mask_class=mask_class, return_stg1=True
    )

    assert fake_b.shape == image.shape
    assert fake_b_stg1.shape == image.shape
    assert fake_b.shape[1] == 3
    assert fake_b_stg1.shape[1] == 3


def test_mat_model_mask_conversion_and_eval_determinism(tmp_path):
    opt = make_mat_opt(tmp_path)
    opt.isTrain = False

    model = MATModel(opt, rank=0)
    batch = make_batch(path="/tmp/mat_sample.png")
    model.set_input(batch)

    expected_hole_mask = batch["B_label_mask"].squeeze(1).float()
    torch.testing.assert_close(model.mask.squeeze(1), expected_hole_mask)
    torch.testing.assert_close(model.mask_keep.squeeze(1), 1.0 - expected_hole_mask)

    model.inference(1, offset=0)
    first_fake = model.fake_B.clone()
    first_stg1 = model.fake_B_stg1.clone()
    first_seeds = list(model.eval_seeds)
    visuals = model.get_current_visuals(1)

    assert list(visuals[0].keys()) == ["gt_image_0", "y_t_0", "mask_0", "output_0"]
    assert visuals[0]["mask_0"].ndim == 3
    torch.testing.assert_close(visuals[0]["gt_image_0"], model.real_B[:1])
    torch.testing.assert_close(visuals[0]["y_t_0"], model.real_A[:1])
    torch.testing.assert_close(visuals[0]["output_0"], model.fake_B[:1])

    model.inference(1, offset=0)
    torch.testing.assert_close(model.fake_B, first_fake)
    torch.testing.assert_close(model.fake_B_stg1, first_stg1)
    assert model.eval_seeds == first_seeds

    second_batch = make_batch(path="/tmp/mat_other_sample.png")
    model.set_input(second_batch)
    model.inference(1, offset=0)
    assert model.eval_seeds != first_seeds


def test_mat_model_mask_class_conditioning_map(tmp_path):
    opt = make_mat_opt(
        tmp_path, alg_mat_mask_class_conditioning=True, f_s_semantic_nclasses=4
    )
    model = MATModel(opt, rank=0)
    batch = make_batch(b_mask_value=2)

    model.set_input(batch)

    expected_mask_class = torch.zeros_like(model.mask)
    expected_mask_class[:, :, 64:128, 96:160] = 2.0
    torch.testing.assert_close(model.mask_class, expected_mask_class)

    model.inference(1, offset=0)
    assert model.fake_B.shape[1] == 3
    assert model.fake_B_stg1.shape[1] == 3


class TinyMapping(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.fc = nn.Linear(z_dim, w_dim)
        self.register_buffer("w_avg", torch.zeros(w_dim))

    def forward(
        self,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        skip_w_avg_update=False,
    ):
        x = self.fc(z.float())
        if self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, 0.995))
        x = x.unsqueeze(1)
        if truncation_psi != 1:
            x = self.w_avg.view(1, 1, -1).lerp(x, truncation_psi)
        return x


class TinySynthesis(nn.Module):
    def __init__(self, img_channels):
        super().__init__()
        self.num_layers = 1
        self.conv = nn.Conv2d(img_channels + 1, img_channels, kernel_size=1)

    def forward(
        self,
        images_in,
        masks_in,
        ws,
        mask_class=None,
        noise_mode="random",
        return_stg1=False,
    ):
        style = ws[:, 0, :].mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        stage1 = torch.tanh(self.conv(torch.cat([images_in, masks_in], dim=1)) + style)
        final = stage1 * (1 - masks_in) + images_in * masks_in
        if return_stg1:
            return final, stage1
        return final


class TinyGenerator(nn.Module):
    def __init__(
        self,
        z_dim,
        c_dim,
        w_dim,
        img_resolution,
        img_channels,
        synthesis_kwargs=None,
        mapping_kwargs=None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.mapping = TinyMapping(z_dim=z_dim, w_dim=w_dim)
        self.synthesis = TinySynthesis(img_channels=img_channels)

    def forward(
        self,
        images_in,
        masks_in,
        z,
        c,
        mask_class=None,
        truncation_psi=1,
        truncation_cutoff=None,
        skip_w_avg_update=False,
        noise_mode="random",
        return_stg1=False,
    ):
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            skip_w_avg_update=skip_w_avg_update,
        )
        return self.synthesis(
            images_in,
            masks_in,
            ws,
            mask_class=mask_class,
            noise_mode=noise_mode,
            return_stg1=return_stg1,
        )


class TinyDiscriminator(nn.Module):
    def __init__(
        self,
        c_dim,
        img_resolution,
        img_channels,
        channel_base=32768,
        channel_max=512,
        channel_decay=1,
        cmap_dim=None,
        activation="lrelu",
        mbstd_group_size=4,
        mbstd_num_channels=1,
    ):
        super().__init__()
        self.final_head = nn.Conv2d(img_channels + 1, 1, kernel_size=1)
        self.stage1_head = nn.Conv2d(img_channels + 1, 1, kernel_size=1)

    def forward(self, images_in, masks_in, images_stg1, c):
        logits = self.final_head(torch.cat([images_in, masks_in], dim=1)).mean(
            dim=(2, 3)
        )
        logits_stg1 = self.stage1_head(torch.cat([images_stg1, masks_in], dim=1)).mean(
            dim=(2, 3)
        )
        return logits, logits_stg1


class TinyPerceptualLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, gt):
        return torch.mean(torch.abs(x - gt)), None


def test_mat_model_training_step_with_lazy_r1_and_ema(monkeypatch, tmp_path):
    monkeypatch.setattr(mat_model_module, "MATGenerator", TinyGenerator)
    monkeypatch.setattr(mat_model_module, "MATDiscriminator", TinyDiscriminator)
    monkeypatch.setattr(mat_model_module, "MATPerceptualLoss", TinyPerceptualLoss)

    opt = make_mat_opt(
        tmp_path,
        alg_mat_d_reg_every=2,
        alg_mat_mask_class_conditioning=True,
        f_s_semantic_nclasses=3,
    )
    model = MATModel(opt, rank=0)
    batch = make_batch(b_mask_value=2)

    model.set_input(batch)
    model.optimize_parameters()

    assert model.loss_D_r1.item() == pytest.approx(0.0)
    assert torch.isfinite(model.loss_G_tot)
    assert torch.isfinite(model.loss_D_tot)
    assert hasattr(model, "netG_A_ema")
    assert model.mask_class is not None

    model.set_input(batch)
    model.optimize_parameters()

    assert model.loss_D_r1.item() > 0
    assert model.loss_D_r1_stg1.item() > 0
    assert model.loss_G_l1.item() >= 0
