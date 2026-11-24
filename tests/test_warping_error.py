import csv
import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from scripts import eval_warping_error as warp


def write_frames(folder, count=3):
    for index in range(count):
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
            folder / f"{index:06d}_generated.png"
        )
        (folder / f"{index:06d}_orig_bbox.json").write_text(json.dumps([5, 6, 15, 16]))


def test_existing_synthetic_metric_checks():
    # Known translation, color drift, inconsistent flow, bounds, and empty masks.
    warp.self_test()


@pytest.mark.parametrize("suffix", ["_generated.png", "_orig_bbox.json"])
def test_missing_frame_or_annotation_is_rejected(tmp_path, suffix):
    write_frames(tmp_path)
    (tmp_path / f"000001{suffix}").unlink()
    with pytest.raises(FileNotFoundError):
        warp.discover_frames(tmp_path, 0, 3)


@pytest.mark.parametrize(
    "bbox", [[-1, 0, 5, 5], [5, 5, 5, 10], [0, 0, 33, 10], [0, 0, float("nan"), 10]]
)
def test_invalid_annotation_is_rejected(tmp_path, bbox):
    write_frames(tmp_path)
    (tmp_path / "000001_orig_bbox.json").write_text(json.dumps(bbox))
    with pytest.raises(ValueError):
        warp.discover_frames(tmp_path, 0, 3)


def test_dry_run_does_not_initialize_raft(tmp_path, monkeypatch):
    write_frames(tmp_path)
    output = tmp_path / "evaluation"

    def unexpected_model(*args, **kwargs):
        pytest.fail("Dry run must not load RAFT weights")

    monkeypatch.setattr(warp, "init_raft", unexpected_model)
    assert (
        warp.main(
            [
                "--input-dir",
                str(tmp_path),
                "--output-dir",
                str(output),
                "--num-frames",
                "3",
                "--dry-run",
            ]
        )
        == 0
    )
    config = json.loads((output / "config.json").read_text())
    assert config["dry_run"] is True
    assert config["fixed_crop_xyxy"] == [2, 3, 18, 19]
    assert config["raft_padding_right_bottom"] == [112, 112]
    assert not (output / "summary.csv").exists()


def test_mocked_cli_writes_scores_and_summary(tmp_path, monkeypatch):
    write_frames(tmp_path)
    output = tmp_path / "evaluation"

    def initialize(size, device):
        def zero_flow(a, b):
            assert a.shape[-2:] == (128, 128)
            return [torch.zeros((1, 2, *a.shape[-2:]), device=device)]

        return zero_flow, lambda a, b: (a.float(), b.float()), "mock"

    monkeypatch.setattr(warp, "init_raft", initialize)
    assert (
        warp.main(
            [
                "--input-dir",
                str(tmp_path),
                "--output-dir",
                str(output),
                "--num-frames",
                "3",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    with (output / "per_transition.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(r["frame_t"], r["frame_t1"]) for r in rows] == [("0", "1"), ("1", "2")]
    with (output / "summary.csv").open() as stream:
        summary = next(csv.DictReader(stream))
    assert int(summary["transitions"]) == 2
    assert int(summary["invalid_transitions"]) == 0
    assert float(summary["mean_roi_warp_mse_rgb_sum"]) == 0
    assert float(summary["mean_roi_total_valid_coverage"]) == 1


def test_summary_averages_valid_transitions_and_reports_invalid(tmp_path):
    write_frames(tmp_path)
    frames, wh = warp.discover_frames(tmp_path, 0, 3)
    crop = warp.fixed_crop([frame.bbox for frame in frames], wh)
    image = torch.zeros(3, 16, 16)
    mask = torch.ones(16, 16, dtype=torch.bool)
    flow = torch.zeros(2, 16, 16)
    valid, _ = warp.evaluate_pair(image, image + 0.2, mask, mask, flow, flow)
    invalid, _ = warp.evaluate_pair(image, image, mask, mask & False, flow, flow)
    summary = warp.summarize([valid, invalid], frames, crop, tmp_path, "mock")
    assert summary["valid_transitions"] == 1
    assert summary["invalid_transitions"] == 1
    assert summary["mean_roi_warp_mse_rgb_sum"] == pytest.approx(0.12)
    empty = warp.summarize([invalid], frames, crop, tmp_path, "mock")
    assert math.isnan(empty["mean_roi_warp_mse_rgb_sum"])
