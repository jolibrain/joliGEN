import math
from types import SimpleNamespace

import pytest
import torch

import models.base_model as base_model_module
from models.base_model import BaseModel
from util.util import MAX_INT


class B2BMetricModel:
    def __init__(self, use_mask):
        self.opt = SimpleNamespace(
            train_nb_img_max_fid=MAX_INT,
            model_type="b2b",
            alg_b2b_metric_mask=use_mask,
            isTrain=True,
            test_batch_size=1,
            G_netG="vit_vid",
            data_direction="AtoB",
            train_metrics_list=["PSNR"],
        )
        self.use_temporal = False
        self.use_inception = False
        self.visual_names = []

    def _is_b2b_validation_loss_enabled(self):
        return False

    _compute_masked_psnr = staticmethod(BaseModel._compute_masked_psnr)

    def _compute_metrics(self, *args, **kwargs):
        return BaseModel._compute_metrics(self, *args, **kwargs)

    def set_input(self, data):
        self.gt_image = data["real"]
        self.real_B = self.gt_image
        self.mask = data["mask"]
        self.step_outputs = data["step_outputs"]

    def inference(self, _batch_size, offset=0):
        del offset
        self.outputs_per_step = self.step_outputs
        self.fake_B = torch.cat(list(self.outputs_per_step.values()), dim=0)
        self.gt_image = self.gt_image.repeat(len(self.outputs_per_step), 1, 1, 1, 1)


def _metric_batch():
    real = torch.zeros(1, 2, 3, 2, 2)
    mask = torch.zeros(1, 2, 1, 2, 2)
    mask[:, 1, :, 0, 0] = 1

    step_2 = torch.ones_like(real)
    step_5 = torch.ones_like(real)
    step_5[:, 1, :, 0, 0] = 0.5
    return {
        "real": real,
        "mask": mask,
        "step_outputs": {2: step_2, 5: step_5},
    }


def _patch_full_crop_metrics(monkeypatch, psnr_value=42.0):
    monkeypatch.setattr(
        base_model_module, "ssim", lambda _real, _fake: torch.tensor(1.0)
    )

    def fake_psnr(real, _fake, reduction="mean", **_kwargs):
        if reduction == "none":
            return torch.full((real.shape[0],), psnr_value)
        return torch.tensor(psnr_value)

    monkeypatch.setattr(base_model_module, "psnr", fake_psnr)


def test_masked_psnr_uses_only_masked_pixels_and_skips_empty_masks():
    real = torch.zeros(3, 3, 2, 2)
    fake = torch.ones_like(real)
    mask = torch.zeros(3, 1, 2, 2)

    mask[0, :, 0, 0] = 1
    fake[0, :, 0, 0] = 0.5
    mask[1] = 1
    fake[1] = 0.5

    result = BaseModel._compute_masked_psnr(real, fake, mask)

    assert result.item() == pytest.approx(10 * math.log10(4))


def test_b2b_metric_mask_applies_to_global_and_per_step_psnr(monkeypatch):
    _patch_full_crop_metrics(monkeypatch)
    model = B2BMetricModel(use_mask=True)

    BaseModel.compute_metrics_test(model, [(_metric_batch(),)], 1, 1000)

    assert model.psnr_test_.item() == pytest.approx(9.030899, rel=1e-5)
    assert model.psnr_step_results[""][2] == pytest.approx(6.0206, rel=1e-5)
    assert model.psnr_step_results[""][5] == pytest.approx(12.0412, rel=1e-5)


def test_b2b_metric_mask_disabled_keeps_full_crop_psnr(monkeypatch):
    _patch_full_crop_metrics(monkeypatch)
    model = B2BMetricModel(use_mask=False)

    BaseModel.compute_metrics_test(model, [(_metric_batch(),)], 1, 1000)

    assert model.psnr_test_.item() == pytest.approx(42.0)
    assert model.psnr_step_results[""] == {2: 42.0, 5: 42.0}
