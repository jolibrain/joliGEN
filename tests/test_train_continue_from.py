import json
from types import SimpleNamespace

import pytest
import torch

from models.base_model import BaseModel
from options.train_options import TrainOptions
from train import save_finetune_source_metadata


def _opt(tmp_path, **overrides):
    values = {
        "gpu_ids": [],
        "isTrain": True,
        "with_amp": False,
        "use_cuda": False,
        "checkpoints_dir": str(tmp_path),
        "name": "target_run",
        "data_preprocess": "resize_and_crop",
        "data_online_context_pixels": 0,
        "G_netG": "mobile_resnet_attn",
        "train_pool_size": 0,
        "train_metrics_list": [],
        "output_display_G_attention_masks": False,
        "train_continue": False,
        "train_continue_from": "",
        "train_load_iter": 0,
        "train_epoch": "latest",
        "train_finetune": False,
        "train_lr_policy": "linear",
        "train_epoch_count": 1,
        "train_n_epochs": 1,
        "train_n_epochs_decay": 0,
        "model_prior_321_backwardcompatibility": False,
        "alg_diffusion_ddpm_cm_ft": False,
        "model_load_no_strictness": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TinyModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt, rank=0)
        self.model_names = ["G_A"]
        self.netG_A = torch.nn.Linear(2, 2)

    def set_input(self, input):
        pass

    def forward(self):
        pass

    def optimize_parameters(self):
        pass


def _save_source_checkpoint(source_dir, suffix, fill_value):
    source_dir.mkdir(parents=True, exist_ok=True)
    net = torch.nn.Linear(2, 2)
    with torch.no_grad():
        net.weight.fill_(fill_value)
        net.bias.fill_(fill_value)
    torch.save(net.state_dict(), source_dir / f"{suffix}_net_G_A.pth")


def test_train_continue_from_loads_weights_from_source_dir(tmp_path):
    source_dir = tmp_path / "source_run"
    _save_source_checkpoint(source_dir, "latest", fill_value=3.0)

    opt = _opt(tmp_path, train_continue_from=str(source_dir))
    model = TinyModel(opt)

    with torch.no_grad():
        model.netG_A.weight.fill_(0.0)
        model.netG_A.bias.fill_(0.0)

    model.setup(opt)

    assert model.save_dir == str(tmp_path / "target_run")
    assert torch.allclose(
        model.netG_A.weight, torch.full_like(model.netG_A.weight, 3.0)
    )
    assert torch.allclose(model.netG_A.bias, torch.full_like(model.netG_A.bias, 3.0))


def test_train_continue_from_uses_train_load_iter_suffix(tmp_path):
    source_dir = tmp_path / "source_run"
    _save_source_checkpoint(source_dir, "latest", fill_value=1.0)
    _save_source_checkpoint(source_dir, "iter_123", fill_value=5.0)

    opt = _opt(tmp_path, train_continue_from=str(source_dir), train_load_iter=123)
    model = TinyModel(opt)

    model.setup(opt)

    assert torch.allclose(
        model.netG_A.weight, torch.full_like(model.netG_A.weight, 5.0)
    )
    assert torch.allclose(model.netG_A.bias, torch.full_like(model.netG_A.bias, 5.0))


def test_train_continue_loads_from_target_run_when_continue_from_is_empty(tmp_path):
    target_dir = tmp_path / "target_run"
    _save_source_checkpoint(target_dir, "latest", fill_value=7.0)

    opt = _opt(tmp_path, train_continue=True, train_continue_from="")
    model = TinyModel(opt)

    model.setup(opt)

    assert torch.allclose(
        model.netG_A.weight, torch.full_like(model.netG_A.weight, 7.0)
    )
    assert torch.allclose(model.netG_A.bias, torch.full_like(model.netG_A.bias, 7.0))


def test_train_continue_and_train_continue_from_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainOptions().parse_json(
            {
                "dataroot": "unused",
                "train_continue": True,
                "train_continue_from": "/tmp/source_run",
            },
            save_config=False,
            set_device=False,
        )


def test_train_continue_from_metadata_records_source_checkpoint(tmp_path):
    opt = _opt(tmp_path, train_continue_from="source_run", train_load_iter=123)

    save_finetune_source_metadata(opt, "python train.py ...", ["G_A"])

    metadata_path = tmp_path / "target_run" / "finetune_source.json"
    metadata = json.loads(metadata_path.read_text())

    assert metadata["train_continue_from"] == "source_run"
    assert metadata["load_suffix"] == "iter_123"
    assert metadata["checkpoint_files"] == ["source_run/iter_123_net_G_A.pth"]
    assert metadata["command_line"] == "python train.py ..."
