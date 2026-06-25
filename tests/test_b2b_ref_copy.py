import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.b2b_model import B2BModel


class CopyLossNet(nn.Module):
    def __init__(self, raw_x_pred):
        super().__init__()
        self.raw_x_pred = raw_x_pred
        self.calls = []

    def forward(
        self,
        x,
        mask=None,
        x_cond=None,
        label=None,
        use_gt=None,
        ref_idx=None,
        return_x_pred=False,
        return_raw_x_pred=False,
    ):
        self.calls.append(
            {
                "return_x_pred": return_x_pred,
                "return_raw_x_pred": return_raw_x_pred,
            }
        )
        v_pred = torch.zeros_like(x)
        v = torch.zeros_like(x)
        x_pred = torch.zeros_like(x)
        if return_raw_x_pred:
            return v_pred, v, x_pred, self.raw_x_pred.to(x.device)
        if return_x_pred:
            return v_pred, v, x_pred
        return v_pred, v


def _make_b2b_model(lambda_ref_copy=0.0, loss="MSE"):
    model = B2BModel.__new__(B2BModel)
    model.opt = SimpleNamespace(
        alg_b2b_loss=loss,
        alg_b2b_lambda_ref_copy=lambda_ref_copy,
        alg_diffusion_lambda_G=1.0,
        alg_b2b_minsnr=False,
        alg_b2b_loss_masked_region_only=False,
        alg_b2b_perceptual_loss=[""],
    )
    model.num_classes = 1
    model.cond_image = None
    model.mask = None
    model.label_cls = torch.zeros(2, dtype=torch.long)
    model.loss_fn = torch.nn.MSELoss()
    return model


def _make_after_parse_opt(**overrides):
    opt = SimpleNamespace(
        alg_b2b_denoise_timesteps=[1],
        alg_b2b_P_std=0.8,
        alg_b2b_t_eps=5e-2,
        alg_b2b_use_gt_prob=0.1,
        alg_b2b_multi_dataset_class_conditioning=False,
        G_vit_vid_motion_every=0,
        alg_b2b_lambda_ref_copy=0.0,
        alg_b2b_ref_degrade_prob=0.0,
        alg_b2b_ref_degrade_noise_std=0.05,
        alg_b2b_lora=False,
    )
    for key, value in overrides.items():
        setattr(opt, key, value)
    return opt


def test_b2b_global_context_mode_parser_default_is_json_safe():
    parser = argparse.ArgumentParser()
    parser = B2BModel.modify_commandline_options(parser, is_train=True)
    opt = parser.parse_args([])

    assert opt.alg_b2b_global_context_mode == "none"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("alg_b2b_lambda_ref_copy", -1.0, "lambda_ref_copy must be >= 0"),
        ("alg_b2b_ref_degrade_prob", 1.1, "ref_degrade_prob must be in"),
        ("alg_b2b_ref_degrade_noise_std", -0.1, "ref_degrade_noise_std must be >= 0"),
    ],
)
def test_after_parse_validates_reference_copy_and_degradation_options(
    field, value, message
):
    with pytest.raises(ValueError, match=message):
        B2BModel.after_parse(_make_after_parse_opt(**{field: value}))


def test_reference_frame_degradation_only_changes_selected_inputs():
    model = B2BModel.__new__(B2BModel)
    model.opt = SimpleNamespace(
        alg_b2b_ref_degrade_prob=1.0,
        alg_b2b_ref_degrade_noise_std=0.5,
    )
    y_t = torch.zeros(2, 3, 1, 4, 4)
    use_gt = torch.tensor([True, False])
    ref_idx = torch.tensor([1, 2])

    torch.manual_seed(0)
    degraded = model._degrade_b2b_reference_frames(y_t, use_gt, ref_idx)

    assert not torch.equal(degraded[0, 1], y_t[0, 1])
    assert torch.equal(degraded[0, 0], y_t[0, 0])
    assert torch.equal(degraded[0, 2], y_t[0, 2])
    assert torch.equal(degraded[1], y_t[1])
    assert degraded.min() >= -1.0
    assert degraded.max() <= 1.0


def test_reference_copy_loss_uses_only_selected_reference_frames():
    model = _make_b2b_model(lambda_ref_copy=2.0)
    target = torch.ones(2, 3, 1, 2, 2)
    raw_x_pred = target.clone()
    raw_x_pred[0, 1] = 0.0
    raw_x_pred[1] = 0.0
    use_gt = torch.tensor([True, False])
    ref_idx = torch.tensor([1, 2])

    loss = model._b2b_reference_copy_loss(raw_x_pred, target, use_gt, ref_idx)

    assert torch.isclose(loss, torch.tensor(1.0))


def test_compute_b2b_loss_adds_weighted_reference_copy_loss():
    model = _make_b2b_model(lambda_ref_copy=2.0)
    model.gt_image = torch.ones(2, 3, 1, 2, 2)
    raw_x_pred = model.gt_image.clone()
    raw_x_pred[0, 1] = 0.0
    raw_x_pred[1] = 0.0
    model.netG_A = CopyLossNet(raw_x_pred)
    model.use_gt = torch.tensor([True, False])
    model.ref_idx = torch.tensor([1, 2])

    model.compute_b2b_loss()

    assert model.netG_A.calls[0]["return_x_pred"]
    assert model.netG_A.calls[0]["return_raw_x_pred"]
    assert torch.isclose(model.loss_G_ref_copy, torch.tensor(2.0))
    assert torch.isclose(model.loss_G_tot, torch.tensor(2.0))


def test_compute_b2b_loss_keeps_raw_copy_path_disabled_by_default():
    model = _make_b2b_model(lambda_ref_copy=0.0)
    model.gt_image = torch.ones(2, 3, 1, 2, 2)
    model.netG_A = CopyLossNet(torch.zeros_like(model.gt_image))
    model.use_gt = torch.tensor([True, False])
    model.ref_idx = torch.tensor([1, 2])

    model.compute_b2b_loss()

    assert not model.netG_A.calls[0]["return_x_pred"]
    assert not model.netG_A.calls[0]["return_raw_x_pred"]
    assert torch.isclose(model.loss_G_tot, torch.tensor(0.0))
