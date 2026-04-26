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
from scripts.gen_multi_dataset_b2b_config import (
    add_test_sets,
    build_multi_dataset_config,
)


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


def test_multi_dataset_list_test_sets_reads_config(tmp_path):
    config_path = tmp_path / "multi.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "a",
                        "dataset_mode": "self_supervised_vid_mask_online",
                        "dataroot": "/data/a",
                    }
                ],
                "test_sets": [
                    {
                        "id": "a",
                        "dataset_name": "a",
                        "dataroot": "/data/a",
                        "child_test_name": "",
                        "generated": False,
                    },
                    {
                        "id": "a__hard",
                        "dataset_name": "a",
                        "dataroot": "/data/a",
                        "child_test_name": "hard",
                        "generated": False,
                    },
                ],
            }
        )
    )

    opt = make_opt(config_path)

    assert data_module.list_test_sets(opt) == ["a", "a__hard"]


def test_multi_dataset_test_phase_delegates_to_named_child_test_set(
    tmp_path, monkeypatch
):
    FakeChildDataset.instances = []
    monkeypatch.setattr(data_module, "find_dataset_using_name", lambda _: FakeChildDataset)
    config_path = tmp_path / "multi.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "a",
                        "dataset_mode": "self_supervised_vid_mask_online",
                        "dataroot": "/data/a",
                    },
                    {
                        "name": "b",
                        "dataset_mode": "self_supervised_vid_mask_online",
                        "dataroot": "/data/b",
                    },
                ],
                "test_sets": [
                    {
                        "id": "b__hard",
                        "dataset_name": "b",
                        "dataroot": "/generated/b",
                        "child_test_name": "hard",
                        "generated": True,
                    }
                ],
            }
        )
    )

    dataset = MultiDatasetDataset(make_opt(config_path), "test", "b__hard")
    sample = dataset[1]

    assert len(dataset) == 3
    assert len(FakeChildDataset.instances) == 1
    assert FakeChildDataset.instances[0].phase == "test"
    assert FakeChildDataset.instances[0].name == "hard"
    assert FakeChildDataset.instances[0].opt.dataroot == "/generated/b"
    assert sample["dataset_name"] == "b"
    assert sample["dataset_index"] == 1
    assert sample["dataset_test_name"] == "b__hard"


def test_multi_dataset_test_phase_rejects_unknown_test_set(tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "find_dataset_using_name", lambda _: FakeChildDataset)
    config_path = tmp_path / "multi.json"
    config_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "name": "a",
                        "dataset_mode": "self_supervised_vid_mask_online",
                        "dataroot": "/data/a",
                    }
                ],
                "test_sets": [
                    {
                        "id": "a",
                        "dataset_name": "a",
                        "dataroot": "/data/a",
                        "child_test_name": "",
                        "generated": False,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="unknown multi_dataset test set"):
        MultiDatasetDataset(make_opt(config_path), "test", "missing")


def test_multi_dataset_generator_discovers_existing_test_sets(tmp_path):
    dataroot = tmp_path / "dataset"
    (dataroot / "testA").mkdir(parents=True)
    (dataroot / "testAhard").mkdir(parents=True)
    (dataroot / "testA" / "paths.txt").write_text("img0.png mask0.png\n")
    (dataroot / "testAhard" / "paths.txt").write_text("img1.png mask1.png\n")

    config = {
        "datasets": [
            {
                "name": "dataset a",
                "dataset_mode": "self_supervised_vid_mask_online",
                "dataroot": str(dataroot),
                "weight": 1.0,
                "overrides": {},
            }
        ]
    }

    add_test_sets(config, tmp_path / "out", SimpleNamespace())

    assert config["test_sets"] == [
        {
            "id": "dataset_a",
            "dataset_name": "dataset a",
            "dataroot": str(dataroot),
            "child_test_name": "",
            "generated": False,
        },
        {
            "id": "dataset_a__hard",
            "dataset_name": "dataset a",
            "dataroot": str(dataroot),
            "child_test_name": "hard",
            "generated": False,
        },
    ]


def test_multi_dataset_generator_creates_true_holdout(tmp_path):
    dataroot = tmp_path / "dataset"
    (dataroot / "trainA").mkdir(parents=True)
    lines = [
        f"vid/frame_{index:03d}.png masks/frame_{index:03d}.png"
        for index in range(6)
    ]
    (dataroot / "trainA" / "paths.txt").write_text("\n".join(lines) + "\n")

    args = SimpleNamespace(
        data_temporal_number_frames=2,
        data_temporal_frame_step=1,
        auto_test_samples=2,
        auto_test_seed=7,
    )
    config = {
        "datasets": [
            {
                "name": "dataset",
                "dataset_mode": "self_supervised_vid_mask_online",
                "dataroot": str(dataroot),
                "weight": 1.0,
                "overrides": {},
            }
        ]
    }

    add_test_sets(config, tmp_path / "out", args)

    generated_root = tmp_path / "out" / "generated_test_sets" / "dataset"
    train_lines = (generated_root / "trainA" / "paths.txt").read_text().splitlines()
    test_lines = (generated_root / "testA" / "paths.txt").read_text().splitlines()

    assert config["datasets"][0]["dataroot"] == str(generated_root)
    assert config["test_sets"] == [
        {
            "id": "dataset",
            "dataset_name": "dataset",
            "dataroot": str(generated_root),
            "child_test_name": "",
            "generated": True,
        }
    ]
    assert train_lines
    assert test_lines
    assert set(train_lines).isdisjoint(set(test_lines))
    assert all(line.startswith(str(dataroot)) for line in train_lines + test_lines)
