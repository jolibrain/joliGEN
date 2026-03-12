import argparse
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.append(sys.path[0] + "/..")

from models.b2b_model import B2BModel
from models.modules.perceptual_dino import DINOPerceptualLoss


class FakeDINOBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 3
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward_features(self, img):
        pooled = F.adaptive_avg_pool2d(img * self.scale, (2, 2))
        tokens = pooled.flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": F.normalize(tokens, dim=-1)}


def test_b2b_parser_accepts_dino_perceptual_loss():
    parser = argparse.ArgumentParser()
    parser = B2BModel.modify_commandline_options(parser, is_train=True)

    opt = parser.parse_args(
        [
            "--alg_b2b_perceptual_loss",
            "DINO",
            "--alg_b2b_dino_model",
            "dinov2_vitb14_reg",
            "--alg_b2b_dino_resize_resolution",
            "112",
        ]
    )

    assert opt.alg_b2b_perceptual_loss == ["DINO"]
    assert opt.alg_b2b_dino_model == "dinov2_vitb14_reg"
    assert opt.alg_b2b_dino_resize_resolution == 112


def test_dino_perceptual_loss_zero_for_identical_inputs(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: FakeDINOBackbone())
    criterion = DINOPerceptualLoss(resize_resolution=16)

    target = torch.ones(2, 3, 8, 8)
    pred = target.clone().requires_grad_(True)

    loss = criterion(target, pred)

    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-6)
    loss.backward()
    assert pred.grad is not None


def test_dino_perceptual_loss_positive_for_different_inputs(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: FakeDINOBackbone())
    criterion = DINOPerceptualLoss(resize_resolution=16)

    target = torch.ones(2, 3, 8, 8)
    pred = -torch.ones(2, 3, 8, 8)

    loss = criterion(target, pred)

    assert loss > 0.5


def test_dino_perceptual_loss_rejects_non_rgb_inputs(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: FakeDINOBackbone())
    criterion = DINOPerceptualLoss(resize_resolution=16)

    with pytest.raises(ValueError, match="3-channel"):
        criterion(torch.ones(1, 1, 8, 8), torch.ones(1, 1, 8, 8))


def test_b2b_compute_perceptual_losses_includes_dino():
    class ConstantCriterion(torch.nn.Module):
        def forward(self, target, pred):
            return torch.tensor(0.25, device=target.device)

    model = B2BModel.__new__(B2BModel)
    model.opt = SimpleNamespace(alg_b2b_perceptual_loss=["DINO"])
    model.criterionDINO = ConstantCriterion()

    lpips_loss, dists_loss, dino_loss = B2BModel._compute_perceptual_losses(
        model,
        torch.ones(1, 3, 8, 8),
        torch.zeros(1, 3, 8, 8),
    )

    assert lpips_loss.item() == 0.0
    assert dists_loss.item() == 0.0
    assert dino_loss.item() == pytest.approx(0.25)
