import argparse
from types import SimpleNamespace

import pytest
import torch

from models.b2b_model import B2BModel


class ConstantLoss(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, target, pred):
        return pred.new_tensor(self.value)


class RecordingFDLLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, target, pred):
        self.calls.append((target.detach().clone(), pred.detach().clone()))
        return (pred - target).mean(dim=(1, 2, 3))


def test_b2b_perceptual_loss_options_accept_fdl():
    parser = argparse.ArgumentParser()
    parser = B2BModel.modify_commandline_options(parser, is_train=True)

    opt = parser.parse_args(["--alg_b2b_perceptual_loss", "FDL"])

    assert opt.alg_b2b_perceptual_loss == ["FDL"]


def test_b2b_perceptual_loss_options_reject_unknown_value():
    parser = argparse.ArgumentParser()
    parser = B2BModel.modify_commandline_options(parser, is_train=True)

    with pytest.raises(SystemExit):
        parser.parse_args(["--alg_b2b_perceptual_loss", "UNKNOWN"])


def test_compute_perceptual_losses_combines_fdl_with_existing_losses():
    model = B2BModel.__new__(B2BModel)
    model.opt = SimpleNamespace(alg_b2b_perceptual_loss=["LPIPS", "DISTS", "FDL"])
    model.criterionLPIPS = ConstantLoss(2.0)
    model.criterionDISTS = ConstantLoss(3.0)
    model.criterionFDL = RecordingFDLLoss()

    target = torch.zeros(2, 1, 4, 4)
    pred = torch.ones(2, 1, 4, 4)

    lpips_loss, dists_loss, fdl_loss = model._compute_perceptual_losses(target, pred)

    assert lpips_loss.item() == pytest.approx(2.0)
    assert dists_loss.item() == pytest.approx(3.0)
    assert fdl_loss.item() == pytest.approx(0.5)
    assert len(model.criterionFDL.calls) == 1
    fdl_target, fdl_pred = model.criterionFDL.calls[0]
    assert fdl_target.shape == (2, 3, 4, 4)
    assert fdl_pred.shape == (2, 3, 4, 4)
    assert torch.all(fdl_target == 0.5)
    assert torch.all(fdl_pred == 1.0)

    lambda_perceptual = 4.0
    combined = lambda_perceptual * (lpips_loss + dists_loss + fdl_loss)
    assert combined.item() == pytest.approx(22.0)
