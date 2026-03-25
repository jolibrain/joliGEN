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


def make_video_batch(path="/tmp/sample.png", num_frames=3, mask_values=None):
    real_a = torch.randn(1, num_frames, 3, 256, 256)
    real_b = torch.randn(1, num_frames, 3, 256, 256)
    a_mask = torch.zeros(1, num_frames, 1, 256, 256)
    b_mask = torch.zeros(1, num_frames, 1, 256, 256)

    if mask_values is None:
        mask_values = list(range(1, num_frames + 1))

    for frame_idx, mask_value in enumerate(mask_values):
        a_mask[:, frame_idx, :, 64:128, 96:160] = mask_value
        b_mask[:, frame_idx, :, 64:128, 96:160] = mask_value

    return {
        "A": real_a,
        "B": real_b,
        "A_ref": real_a[:, 0],
        "B_ref": real_b[:, 0],
        "A_img_paths": [path],
        "A_label_mask": a_mask,
        "A_ref_label_mask": a_mask[:, 0],
        "B_label_mask": b_mask,
        "B_ref_label_mask": b_mask[:, 0],
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


def test_mat_motion_parse_defaults(tmp_path):
    opt = make_mat_opt(
        tmp_path,
        alg_mat_motion=True,
        data_dataset_mode="self_supervised_vid_mask_online",
        data_temporal_number_frames=3,
    )

    assert opt.alg_mat_motion is True
    assert opt.alg_mat_motion_max_frames >= 3


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {
                "alg_mat_motion": True,
                "data_dataset_mode": "self_supervised_labeled_mask_online",
                "data_temporal_number_frames": 3,
            },
            "self_supervised_vid_mask_online",
        ),
        (
            {
                "alg_mat_motion": True,
                "data_dataset_mode": "self_supervised_vid_mask_online",
                "data_temporal_number_frames": 1,
            },
            "data_temporal_number_frames >= 2",
        ),
    ],
)
def test_mat_motion_parse_rejects_invalid_combinations(tmp_path, overrides, match):
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


def test_vendored_mat_generator_motion_shapes():
    image = torch.randn(1, 2, 3, 256, 256)
    mask_keep = (torch.rand(1, 2, 1, 256, 256) > 0.5).float()
    z = torch.randn(1, 512)

    net_g = Generator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=256,
        img_channels=3,
        synthesis_kwargs={"motion_enabled": True, "motion_max_frames": 4},
    )

    fake_b, fake_b_stg1 = net_g(image, mask_keep, z, None, return_stg1=True)

    assert fake_b.shape == (1, 3, 256, 256)
    assert fake_b_stg1.shape == (1, 3, 256, 256)
    assert torch.isfinite(fake_b).all()


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
    def __init__(self, img_channels, motion_enabled=False, mask_class_channels=0):
        super().__init__()
        self.num_layers = 1
        self.motion_enabled = motion_enabled
        self.mask_class_channels = mask_class_channels
        self.first_stage = nn.Conv2d(
            img_channels + 1 + mask_class_channels, img_channels, kernel_size=1
        )
        self.enc = nn.Conv2d(
            img_channels + 1 + mask_class_channels, img_channels, kernel_size=1
        )
        self.motion_module = (
            nn.Conv3d(img_channels, img_channels, kernel_size=1)
            if motion_enabled
            else nn.Identity()
        )
        self.to_square = nn.Linear(1, 1)
        self.to_style = nn.Conv2d(img_channels, img_channels, kernel_size=1)
        self.dec = nn.Conv2d(img_channels, img_channels, kernel_size=1)

    def forward(
        self,
        images_in,
        masks_in,
        ws,
        mask_class=None,
        noise_mode="random",
        return_stg1=False,
    ):
        del noise_mode

        style = ws[:, 0, :].mean(dim=1, keepdim=True)
        if images_in.ndim == 5:
            batch_size, num_frames = images_in.shape[:2]
            images_flat = images_in.reshape(
                batch_size * num_frames, *images_in.shape[2:]
            )
            masks_flat = masks_in.reshape(batch_size * num_frames, *masks_in.shape[2:])
            mask_class_flat = None
            if self.mask_class_channels > 0 and mask_class is not None:
                mask_class_flat = mask_class.reshape(
                    batch_size * num_frames, *mask_class.shape[2:]
                )
            style_flat = style.repeat_interleave(num_frames, dim=0).view(-1, 1, 1, 1)
            first_stage_inputs = [images_flat, masks_flat]
            enc_inputs = []
            if self.mask_class_channels > 0:
                if mask_class_flat is None:
                    mask_class_flat = torch.zeros(
                        batch_size * num_frames,
                        self.mask_class_channels,
                        images_in.shape[3],
                        images_in.shape[4],
                        device=images_in.device,
                        dtype=images_in.dtype,
                    )
                first_stage_inputs.append(mask_class_flat)

            stage1 = torch.tanh(
                self.first_stage(torch.cat(first_stage_inputs, dim=1)) + style_flat
            )
            enc_inputs = [stage1, masks_flat]
            if self.mask_class_channels > 0:
                enc_inputs.append(mask_class_flat)
            encoded = torch.tanh(self.enc(torch.cat(enc_inputs, dim=1))).reshape(
                batch_size,
                num_frames,
                images_in.shape[2],
                images_in.shape[3],
                images_in.shape[4],
            )

            if self.motion_enabled:
                encoded = self.motion_module(encoded.permute(0, 2, 1, 3, 4)).permute(
                    0, 2, 1, 3, 4
                )

            current_encoded = encoded[:, -1]
            stage1 = stage1.reshape(batch_size, num_frames, *stage1.shape[1:])[:, -1]
            current_images = images_in[:, -1]
            current_masks = masks_in[:, -1]
        else:
            first_stage_inputs = [images_in, masks_in]
            enc_inputs = []
            if self.mask_class_channels > 0:
                if mask_class is None:
                    mask_class = torch.zeros(
                        images_in.shape[0],
                        self.mask_class_channels,
                        images_in.shape[2],
                        images_in.shape[3],
                        device=images_in.device,
                        dtype=images_in.dtype,
                    )
                first_stage_inputs.append(mask_class)
            stage1 = torch.tanh(
                self.first_stage(torch.cat(first_stage_inputs, dim=1))
                + style.view(-1, 1, 1, 1)
            )
            enc_inputs = [stage1, masks_in]
            if self.mask_class_channels > 0:
                enc_inputs.append(mask_class)
            current_encoded = torch.tanh(self.enc(torch.cat(enc_inputs, dim=1)))
            current_images = images_in
            current_masks = masks_in

        style_bias = self.to_square(style[:, :1]).view(-1, 1, 1, 1)
        styled = torch.tanh(self.to_style(current_encoded) + style_bias)
        final = torch.tanh(self.dec(styled))
        final = final * (1 - current_masks) + current_images * current_masks
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
        synthesis_kwargs = synthesis_kwargs or {}
        self.synthesis = TinySynthesis(
            img_channels=img_channels,
            motion_enabled=synthesis_kwargs.get("motion_enabled", False),
            mask_class_channels=synthesis_kwargs.get("mask_class_channels", 0),
        )

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


def test_mat_model_motion_training_step_freezes_backbone(monkeypatch, tmp_path):
    monkeypatch.setattr(mat_model_module, "MATGenerator", TinyGenerator)
    monkeypatch.setattr(mat_model_module, "MATDiscriminator", TinyDiscriminator)
    monkeypatch.setattr(mat_model_module, "MATPerceptualLoss", TinyPerceptualLoss)

    opt = make_mat_opt(
        tmp_path,
        alg_mat_motion=True,
        data_dataset_mode="self_supervised_vid_mask_online",
        data_temporal_number_frames=3,
        alg_mat_d_reg_every=2,
        alg_mat_mask_class_conditioning=True,
        f_s_semantic_nclasses=4,
    )
    model = MATModel(opt, rank=0)

    trainable_names = {
        name for name, param in model.netG_A.named_parameters() if param.requires_grad
    }
    assert "mapping.fc.weight" not in trainable_names
    assert "synthesis.first_stage.weight" not in trainable_names
    assert "synthesis.enc.weight" not in trainable_names
    assert "synthesis.motion_module.weight" in trainable_names
    assert "synthesis.to_square.weight" in trainable_names
    assert "synthesis.to_style.weight" in trainable_names
    assert "synthesis.dec.weight" in trainable_names

    batch = make_video_batch(mask_values=[1, 2, 3])
    model.set_input(batch)

    torch.testing.assert_close(model.real_A_seq[:, :-1], batch["B"][:, :-1])
    torch.testing.assert_close(model.real_A_seq[:, -1], batch["A"][:, -1])
    assert torch.count_nonzero(model.mask_seq[:, :-1]) == 0
    assert torch.count_nonzero(1.0 - model.mask_keep_seq[:, :-1]) == 0
    assert torch.count_nonzero(model.mask_class_seq[:, :-1]) == 0

    model.optimize_parameters()

    assert model.real_A_seq.shape == (1, 3, 3, 256, 256)
    assert model.mask_class_seq.shape == (1, 3, 1, 256, 256)
    assert model.fake_B.shape == (1, 3, 256, 256)
    assert model.fake_B_stg1.shape == (1, 3, 256, 256)
    assert model.loss_G_adv_stg1.item() == pytest.approx(0.0)
    assert model.loss_D_fake_stg1.item() == pytest.approx(0.0)
    assert model.loss_D_real_stg1.item() == pytest.approx(0.0)
    assert torch.isfinite(model.loss_G_tot)
    assert torch.isfinite(model.loss_D_tot)


def test_mat_motion_visuals_include_previous_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(mat_model_module, "MATGenerator", TinyGenerator)
    monkeypatch.setattr(mat_model_module, "MATDiscriminator", TinyDiscriminator)
    monkeypatch.setattr(mat_model_module, "MATPerceptualLoss", TinyPerceptualLoss)

    opt = make_mat_opt(
        tmp_path,
        alg_mat_motion=True,
        data_dataset_mode="self_supervised_vid_mask_online",
        data_temporal_number_frames=3,
    )
    opt.isTrain = False
    model = MATModel(opt, rank=0)
    batch = make_video_batch(mask_values=[1, 2, 3])

    model.set_input(batch)
    model.inference(1, offset=0)
    visuals = model.get_current_visuals(1)

    assert list(visuals[0].keys()) == [
        "previous_frame_0",
        "gt_image_0",
        "y_t_0",
        "mask_0",
        "output_0",
    ]
    torch.testing.assert_close(visuals[0]["previous_frame_0"], batch["B"][:, -2])
    torch.testing.assert_close(model.previous_frame, batch["B"][:, -2])


def test_mat_motion_ignores_ref_frame_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(mat_model_module, "MATGenerator", TinyGenerator)
    monkeypatch.setattr(mat_model_module, "MATDiscriminator", TinyDiscriminator)
    monkeypatch.setattr(mat_model_module, "MATPerceptualLoss", TinyPerceptualLoss)

    opt = make_mat_opt(
        tmp_path,
        alg_mat_motion=True,
        data_dataset_mode="self_supervised_vid_mask_online",
        data_temporal_number_frames=3,
    )
    model = MATModel(opt, rank=0)
    batch = make_video_batch(mask_values=[1, 2, 3])
    batch["A_ref"] = torch.full_like(batch["A_ref"], -7.0)
    batch["B_ref"] = torch.full_like(batch["B_ref"], 9.0)

    model.set_input(batch)

    torch.testing.assert_close(model.real_B_seq, batch["B"])
    torch.testing.assert_close(model.real_A_seq[:, :-1], batch["B"][:, :-1])
    torch.testing.assert_close(model.real_A_seq[:, -1], batch["A"][:, -1])
    torch.testing.assert_close(model.previous_frame, batch["B"][:, -2])
    assert not torch.allclose(model.real_B_seq[:, 0], batch["B_ref"])
    assert not torch.allclose(model.previous_frame, batch["B_ref"])


def test_mat_motion_losses_use_last_frame_only(monkeypatch, tmp_path):
    monkeypatch.setattr(mat_model_module, "MATGenerator", TinyGenerator)
    monkeypatch.setattr(mat_model_module, "MATDiscriminator", TinyDiscriminator)
    monkeypatch.setattr(mat_model_module, "MATPerceptualLoss", TinyPerceptualLoss)

    opt = make_mat_opt(
        tmp_path,
        alg_mat_motion=True,
        data_dataset_mode="self_supervised_vid_mask_online",
        data_temporal_number_frames=3,
    )
    model = MATModel(opt, rank=0)
    batch = make_video_batch(mask_values=[1, 1, 1])

    model.set_input(batch)
    torch.testing.assert_close(model.real_A, batch["A"][:, -1])
    torch.testing.assert_close(model.real_B, batch["B"][:, -1])
    torch.testing.assert_close(model.gt_image, batch["B"][:, -1])

    torch.manual_seed(0)
    model._forward_generator_train()
    model.compute_G_loss()
    loss_ref = model.loss_G_l1.detach().clone()

    modified_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    modified_batch["B"][:, :-1] = torch.randn_like(modified_batch["B"][:, :-1]) * 50.0

    model.set_input(modified_batch)
    torch.manual_seed(0)
    model._forward_generator_train()
    model.compute_G_loss()

    torch.testing.assert_close(model.loss_G_l1, loss_ref)


def test_mat_motion_loader_allows_missing_motion_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(mat_model_module, "MATGenerator", TinyGenerator)
    monkeypatch.setattr(mat_model_module, "MATDiscriminator", TinyDiscriminator)
    monkeypatch.setattr(mat_model_module, "MATPerceptualLoss", TinyPerceptualLoss)

    motion_opt = make_mat_opt(
        tmp_path,
        alg_mat_motion=True,
        data_dataset_mode="self_supervised_vid_mask_online",
        data_temporal_number_frames=3,
        alg_mat_mask_class_conditioning=True,
        f_s_semantic_nclasses=4,
    )
    motion_model = MATModel(motion_opt, rank=0)
    non_motion_generator = TinyGenerator(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=256,
        img_channels=3,
    )

    motion_model._load_generator_state_dict_for_motion(
        motion_model.netG_A, non_motion_generator.state_dict()
    )

    loaded_first_stage = motion_model.netG_A.state_dict()[
        "synthesis.first_stage.weight"
    ]
    source_first_stage = non_motion_generator.state_dict()[
        "synthesis.first_stage.weight"
    ]
    torch.testing.assert_close(
        loaded_first_stage[:, : source_first_stage.shape[1]], source_first_stage
    )
    assert (
        torch.count_nonzero(loaded_first_stage[:, source_first_stage.shape[1] :]) == 0
    )

    loaded_enc = motion_model.netG_A.state_dict()["synthesis.enc.weight"]
    source_enc = non_motion_generator.state_dict()["synthesis.enc.weight"]
    torch.testing.assert_close(loaded_enc[:, : source_enc.shape[1]], source_enc)
    assert torch.count_nonzero(loaded_enc[:, source_enc.shape[1] :]) == 0
