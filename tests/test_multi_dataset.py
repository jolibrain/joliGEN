import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data.dataloader import default_collate

sys.path.append(str(Path(__file__).resolve().parents[1]))

import data as data_module
from data.multi_dataset_dataset import MultiDatasetDataset
from scripts.gen_multi_dataset_b2b_config import build_multi_dataset_config


class FakeChildDataset:
    instances = []

    def __init__(self, opt, phase, name=""):
        self.opt = opt
        self.phase = phase
        self.name = name
        self.length = opt.fake_length
        FakeChildDataset.instances.append(self)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        value = float(getattr(self.opt, "fake_value", 0))
        return {
            "A": torch.full((2, 3, 8, 8), value),
            "B": torch.full((2, 3, 8, 8), value + 1),
            "B_label_mask": torch.zeros(2, 1, 8, 8),
            "A_img_paths": f"{self.opt.dataroot}/{index}.png",
        }


class FlakyChildDataset(FakeChildDataset):
    def __getitem__(self, index):
        if index == 0:
            return None
        return super().__getitem__(index)


def make_opt(config_path, **kwargs):
    values = {
        "data_dataset_mode": "multi_dataset",
        "data_multi_dataset_config": str(config_path),
        "dataroot": "/unused",
        "data_direction": "AtoB",
        "model_input_nc": 3,
        "model_output_nc": 3,
        "checkpoints_dir": "/tmp",
        "name": "multi_dataset_test",
        "warning_mode": True,
        "data_image_bits": 8,
        "data_load_size": 8,
        "data_crop_size": 8,
        "data_temporal_number_frames": 2,
        "data_temporal_frame_step": 1,
        "G_netG": "vit_vid",
        "G_vit_num_classes": 1,
        "model_type": "b2b",
        "alg_b2b_mask_as_channel": False,
        "alg_diffusion_cond_image_creation": "y_t",
        "fake_length": 3,
        "fake_value": 0,
        "data_online_creation_crop_size_A": 8,
        "data_online_creation_crop_delta_A": 0,
        "data_online_creation_load_size_A": [],
        "data_online_creation_mask_delta_A": [[]],
        "data_online_creation_mask_delta_A_ratio": [[]],
        "data_online_creation_mask_random_offset_A": [0.0],
        "data_online_creation_mask_square_A": False,
        "data_temporal_num_common_char": -1,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def write_config(tmp_path, datasets):
    config_path = tmp_path / "multi.json"
    config_path.write_text(json.dumps({"datasets": datasets}))
    return config_path


def test_multi_dataset_parses_config_and_applies_child_overrides(tmp_path, monkeypatch):
    FakeChildDataset.instances = []
    monkeypatch.setattr(data_module, "find_dataset_using_name", lambda _: FakeChildDataset)
    config_path = write_config(
        tmp_path,
        [
            {
                "name": "a",
                "dataset_mode": "self_supervised_vid_mask_online",
                "dataroot": "/data/a",
                "overrides": {
                    "data_online_creation_crop_size_A": 12,
                    "data_temporal_num_common_char": 7,
                },
            },
            {
                "name": "b",
                "dataset_mode": "self_supervised_vid_mask_online",
                "dataroot": "/data/b",
                "weight": 3,
                "overrides": {"data_online_creation_crop_delta_A": 2},
            },
        ],
    )

    dataset = MultiDatasetDataset(make_opt(config_path), "train")

    assert len(dataset) == 6
    assert dataset.child_names == ["a", "b"]
    assert dataset.child_weights == [1.0, 3.0]
    assert FakeChildDataset.instances[0].opt.dataroot == "/data/a"
    assert FakeChildDataset.instances[0].opt.data_dataset_mode == "self_supervised_vid_mask_online"
    assert FakeChildDataset.instances[0].opt.data_online_creation_crop_size_A == 12
    assert FakeChildDataset.instances[0].opt.data_temporal_num_common_char == 7
    assert FakeChildDataset.instances[1].opt.dataroot == "/data/b"
    assert FakeChildDataset.instances[1].opt.data_online_creation_crop_delta_A == 2


def test_multi_dataset_rejects_shape_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "find_dataset_using_name", lambda _: FakeChildDataset)
    config_path = write_config(
        tmp_path,
        [
            {
                "name": "bad",
                "dataset_mode": "self_supervised_vid_mask_online",
                "dataroot": "/data/bad",
                "overrides": {"data_load_size": 16},
            }
        ],
    )

    with pytest.raises(ValueError, match="cannot override shape/model option"):
        MultiDatasetDataset(make_opt(config_path), "train")


def test_multi_dataset_samples_weighted_children_and_collates(tmp_path, monkeypatch):
    FakeChildDataset.instances = []
    monkeypatch.setattr(data_module, "find_dataset_using_name", lambda _: FakeChildDataset)
    config_path = write_config(
        tmp_path,
        [
            {
                "name": "low",
                "dataset_mode": "self_supervised_vid_mask_online",
                "dataroot": "/data/low",
                "weight": 0,
                "overrides": {"data_online_creation_crop_size_A": 9},
            },
            {
                "name": "high",
                "dataset_mode": "self_supervised_vid_mask_online",
                "dataroot": "/data/high",
                "weight": 1,
                "overrides": {"data_online_creation_crop_size_A": 10},
            },
        ],
    )

    dataset = MultiDatasetDataset(make_opt(config_path), "train")
    batch = default_collate([dataset[0], dataset[1]])

    assert batch["A"].shape == (2, 2, 3, 8, 8)
    assert batch["B_label_mask"].shape == (2, 2, 1, 8, 8)
    assert batch["dataset_name"] == ["high", "high"]
    assert torch.equal(batch["dataset_index"], torch.tensor([1, 1]))


def test_multi_dataset_retries_none_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "find_dataset_using_name", lambda _: FlakyChildDataset)
    config_path = write_config(
        tmp_path,
        [
            {
                "name": "flaky",
                "dataset_mode": "self_supervised_vid_mask_online",
                "dataroot": "/data/flaky",
            }
        ],
    )

    dataset = MultiDatasetDataset(make_opt(config_path), "train")
    sample = dataset[0]

    assert sample is not None
    assert sample["dataset_name"] == "flaky"


def test_multi_dataset_generator_rejects_manifest_shape_overrides():
    with pytest.raises(ValueError, match="cannot override shape/model option"):
        build_multi_dataset_config(
            [
                {
                    "name": "bad",
                    "dataroot": "/data/bad",
                    "data_load_size": "512",
                }
            ]
        )


def test_multi_dataset_generator_writes_expected_config_shape():
    config = build_multi_dataset_config(
        [
            {
                "name": "a",
                "dataroot": "/data/a",
                "weight": "2",
                "data_online_creation_crop_size_A": "304",
                "data_online_creation_mask_delta_A_ratio": "[[0.2, 0.2]]",
            }
        ]
    )

    assert config == {
        "datasets": [
            {
                "name": "a",
                "dataset_mode": "self_supervised_vid_mask_online",
                "dataroot": "/data/a",
                "weight": 2.0,
                "overrides": {
                    "data_online_creation_crop_size_A": 304,
                    "data_online_creation_mask_delta_A_ratio": [[0.2, 0.2]],
                },
            }
        ]
    }
