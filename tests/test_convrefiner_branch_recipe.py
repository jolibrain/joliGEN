import json

import pytest

from scripts.prepare_b2b_convrefiner_branch import prepare_branch


def source_recipe(tmp_path):
    run = tmp_path / "original"
    run.mkdir()
    for suffix in ["net_G_A", "net_G_A_ema", "net_D_refiner", "refiner_training"]:
        (run / f"5_{suffix}.pth").touch()
    config = {
        "model_type": "b2b",
        "name": "original",
        "checkpoints_dir": str(tmp_path),
        "alg": {
            "b2b_convrefiner_finetune": True,
            "b2b_denoise_timesteps": [2, 5],
            "b2b_convrefiner_gan_weight": 0.02,
        },
        "train": {
            "G_ema": True,
            "G_lr": 1e-4,
            "D_lr": 1e-4,
            "n_epochs": 30,
            "continue_from": "backbone",
        },
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(config))
    return path, config


def test_prepare_full_state_branch_preserves_original(tmp_path):
    source, original = source_recipe(tmp_path)
    output = tmp_path / "branch.json"
    assert prepare_branch(source, output) == output
    config = json.loads(output.read_text())
    assert json.loads(source.read_text()) == original
    assert config["name"] == "original_sampler_aligned_from_5"
    assert config["alg"]["b2b_denoise_timesteps"] == [2, 5]
    assert config["alg"]["b2b_convrefiner_gan_schedule"] == "sampler_aligned"
    assert config["alg"]["b2b_convrefiner_resume_from"] == str(tmp_path / "original")
    assert config["train"]["continue"] is True
    assert config["train"]["continue_from"] == ""
    assert config["train"]["epoch"] == "5"
    assert config["train"]["G_lr"] == original["train"]["G_lr"]
    assert not (tmp_path / config["name"]).exists()
    with pytest.raises(FileExistsError):
        prepare_branch(source, output)
    with pytest.raises(FileExistsError):
        prepare_branch(source, tmp_path / "other.json", name="original")


def test_branch_requires_complete_checkpoint(tmp_path):
    source, _ = source_recipe(tmp_path)
    (tmp_path / "original" / "5_refiner_training.pth").unlink()
    with pytest.raises(FileNotFoundError, match="refiner_training"):
        prepare_branch(source, tmp_path / "branch.json")
    assert not (tmp_path / "branch.json").exists()
