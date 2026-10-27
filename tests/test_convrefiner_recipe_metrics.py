import json
from pathlib import Path

import pytest
import torch

from scripts.prepare_b2b_convrefiner import prepare, recording_key
from util.convrefiner_metrics import RefinerMetrics, acceptance, radial_power


def test_recording_split_is_disjoint_and_does_not_modify_source(tmp_path):
    data = tmp_path / "source_data"
    (data / "trainA").mkdir(parents=True)
    manifest = data / "trainA" / "paths.txt"
    manifest.write_text(
        "\n".join(
            f"images/person-{person}_frame_{frame}.jpg masks/person-{person}_frame_{frame}.png"
            for person in range(10)
            for frame in range(3)
        )
    )
    original = manifest.read_bytes()
    config = {
        "name": "source",
        "model_type": "b2b",
        "dataroot": str(data),
        "train": {},
        "alg": {},
        "data": {
            "dataset_mode": "self_supervised_vid_mask_online",
            "temporal_number_frames": 2,
        },
    }
    source = tmp_path / "train_config.json"
    source.write_text(json.dumps(config))
    output = prepare(source, tmp_path / "recipe")
    split = json.loads((output / "split.json").read_text())
    assert len(split["train"]) == 9 and len(split["test"]) == 1
    assert not set(split["train"]) & set(split["test"])
    assert manifest.read_bytes() == original
    pilot = json.loads((output / "pilot.json").read_text())
    assert pilot["train"]["epoch"] == "320"
    assert pilot["alg"]["b2b_convrefiner_source_ema"]
    datasets = json.loads((output / "datasets.json").read_text())
    assert all(entry["weight"] == 3 for entry in datasets["datasets"])
    with pytest.raises(FileExistsError):
        prepare(source, output)


def test_metrics_identity_and_empty_region_are_well_defined():
    metric = RefinerMetrics()
    target = torch.rand(1, 2, 3, 32, 32) * 2 - 1
    mask = torch.ones(1, 2, 1, 32, 32)
    metric.add(target, target, mask)
    report = metric.result()
    assert report["masked_PSNR"] == 120
    assert report["temporal_difference_MAE"] == 0
    assert report["HF_absolute_deficit_dB"] == 0
    empty = RefinerMetrics()
    empty.add(target, target, mask * 0)
    assert empty.result()["masked_PSNR"] is None


def test_spectral_metric_detects_high_frequency_loss():
    torch.manual_seed(42)
    target = torch.rand(1, 2, 3, 64, 64) * 2 - 1
    blurred = torch.nn.functional.avg_pool2d(target.flatten(0, 1), 3, 1, 1).reshape_as(
        target
    )
    metric = RefinerMetrics()
    metric.add(blurred, target, torch.ones(1, 2, 1, 64, 64))
    assert metric.result()["HF_deficit_dB"] < -5
    baseline = dict(
        masked_PSNR=30,
        LPIPS=0.2,
        DISTS=0.1,
        temporal_difference_MAE=0.1,
        HF_absolute_deficit_dB=6,
    )
    refined = dict(baseline, HF_absolute_deficit_dB=2)
    assert acceptance(baseline, refined)["numeric_pass"]
    assert not acceptance(baseline, dict(refined, temporal_difference_MAE=0.2))[
        "numeric_pass"
    ]
