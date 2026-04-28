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
from models.b2b_model import B2BModel
from scripts.gen_multi_dataset_b2b_config import (
    add_test_sets,
    build_train_config,
    build_multi_dataset_config,
    discover_dataset_roots,
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
        "alg_b2b_multi_dataset_class_conditioning": False,
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


def make_generator_args(**kwargs):
    values = {
        "name": "b2b_multi_dataset",
        "checkpoints_dir": "./checkpoints",
        "gpu_ids": "-1",
        "base_train_config": "",
        "reference_dataroot": "/data/a",
        "coverage": 0.75,
        "step": 16,
        "size": None,
        "weight": 1.0,
        "ignore_categories": ["2"],
        "data_load_size": 256,
        "data_crop_size": 256,
        "data_temporal_number_frames": 2,
        "data_temporal_frame_step": 1,
        "data_num_threads": 8,
        "alg_b2b_multi_dataset_class_conditioning": False,
        "multi_dataset_num_datasets": 1,
        "train_batch_size": 8,
        "train_iter_size": 4,
        "train_n_epochs": 6000,
        "train_n_epochs_decay": 0,
        "train_save_epoch_freq": 1000,
        "train_G_lr": 1e-4,
        "train_metrics_every": 20000,
        "output_print_freq": 200,
        "output_display_freq": 1000,
        "auto_test_samples": 1,
        "auto_test_seed": 7,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def write_bbox_dataset(root, name="ring", sizes=(100, 120, 140, 160)):
    train_dir = root / "trainA"
    bbox_dir = root / name / "bbox"
    train_dir.mkdir(parents=True)
    bbox_dir.mkdir(parents=True)
    lines = []
    for index, size in enumerate(sizes):
        bbox_rel = f"{name}/bbox/frame_{index:03d}.txt"
        (root / bbox_rel).write_text(f"1 0 0 {size} {size}\n2 0 0 400 400\n")
        lines.append(f"{name}/imgs/frame_{index:03d}.png {bbox_rel}")
    (train_dir / "paths.txt").write_text("\n".join(lines) + "\n")
    return root


def test_multi_dataset_parses_config_and_applies_child_overrides(tmp_path, monkeypatch):
    FakeChildDataset.instances = []
    monkeypatch.setattr(
        data_module, "find_dataset_using_name", lambda _: FakeChildDataset
    )
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
    assert (
        FakeChildDataset.instances[0].opt.data_dataset_mode
        == "self_supervised_vid_mask_online"
    )
    assert FakeChildDataset.instances[0].opt.data_online_creation_crop_size_A == 12
    assert FakeChildDataset.instances[0].opt.data_temporal_num_common_char == 7
    assert FakeChildDataset.instances[1].opt.dataroot == "/data/b"
    assert FakeChildDataset.instances[1].opt.data_online_creation_crop_delta_A == 2


def test_multi_dataset_rejects_shape_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data_module, "find_dataset_using_name", lambda _: FakeChildDataset
    )
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


def test_multi_dataset_rejects_too_few_vit_classes_for_dataset_conditioning(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        data_module, "find_dataset_using_name", lambda _: FakeChildDataset
    )
    config_path = write_config(
        tmp_path,
        [
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
    )

    opt = make_opt(
        config_path,
        alg_b2b_multi_dataset_class_conditioning=True,
        G_vit_num_classes=1,
    )

    with pytest.raises(ValueError, match="G_vit_num_classes >= number of datasets"):
        MultiDatasetDataset(opt, "train")


def test_multi_dataset_samples_weighted_children_and_collates(tmp_path, monkeypatch):
    FakeChildDataset.instances = []
    monkeypatch.setattr(
        data_module, "find_dataset_using_name", lambda _: FakeChildDataset
    )
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
    monkeypatch.setattr(
        data_module, "find_dataset_using_name", lambda _: FlakyChildDataset
    )
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


def test_b2b_dataset_conditioning_uses_dataset_index_over_object_labels():
    model = B2BModel.__new__(B2BModel)
    model.opt = SimpleNamespace(alg_b2b_multi_dataset_class_conditioning=True)
    model.device = torch.device("cpu")
    model.y_t = torch.zeros(2, 2, 3, 8, 8)
    model.batch_size = 2
    model.num_classes = 3

    labels = model._select_b2b_labels(
        {
            "dataset_index": torch.tensor([1, 2]),
            "B_label_cls": torch.tensor([0, 0]),
        }
    )

    assert torch.equal(labels, torch.tensor([1, 2]))


def test_b2b_dataset_conditioning_rejects_missing_or_out_of_range_labels():
    model = B2BModel.__new__(B2BModel)
    model.opt = SimpleNamespace(alg_b2b_multi_dataset_class_conditioning=True)
    model.device = torch.device("cpu")
    model.y_t = torch.zeros(2, 2, 3, 8, 8)
    model.batch_size = 2
    model.num_classes = 2

    with pytest.raises(RuntimeError, match="requires multi_dataset samples"):
        model._select_b2b_labels({})

    with pytest.raises(RuntimeError, match="dataset_index labels must be in"):
        model._select_b2b_labels({"dataset_index": torch.tensor([0, 2])})


def test_b2b_dataset_conditioning_requires_multi_dataset_mode():
    opt = SimpleNamespace(
        alg_b2b_denoise_timesteps=[1],
        alg_b2b_P_std=0.8,
        alg_b2b_t_eps=0.05,
        alg_b2b_use_gt_prob=0.1,
        alg_b2b_multi_dataset_class_conditioning=True,
        data_dataset_mode="self_supervised_vid_mask_online",
    )

    with pytest.raises(ValueError, match="requires --data_dataset_mode multi_dataset"):
        B2BModel.after_parse(opt)


def test_multi_dataset_generator_discovers_dataset_roots(tmp_path):
    datasets_root = tmp_path / "datasets"
    dataset_a = write_bbox_dataset(datasets_root / "a")
    dataset_b = write_bbox_dataset(datasets_root / "b")
    (datasets_root / "not_a_dataset").mkdir()

    roots = discover_dataset_roots(
        SimpleNamespace(datasets_root=str(datasets_root), dataset_dirs=None)
    )

    assert roots == [dataset_a.resolve(), dataset_b.resolve()]


def test_multi_dataset_generator_writes_derived_config_shape(tmp_path):
    dataroot = write_bbox_dataset(tmp_path / "dataset")
    args = make_generator_args(size=320, weight=2.0)

    config = build_multi_dataset_config([dataroot], args=args)

    assert config == {
        "datasets": [
            {
                "name": "dataset",
                "dataset_mode": "self_supervised_vid_mask_online",
                "dataroot": str(dataroot),
                "weight": 2.0,
                "overrides": {
                    "data_online_creation_crop_size_A": 320,
                    "data_online_creation_crop_delta_A": 32,
                },
            }
        ]
    }


def test_multi_dataset_generator_ignores_categories_when_deriving_size(tmp_path):
    dataroot = write_bbox_dataset(tmp_path / "dataset", sizes=(100, 100, 100, 100))
    args = make_generator_args(ignore_categories=["2"])

    config = build_multi_dataset_config([dataroot], args=args)

    assert config["datasets"][0]["overrides"]["data_online_creation_crop_size_A"] == 96

    args = make_generator_args(ignore_categories=[])
    config = build_multi_dataset_config([dataroot], args=args)

    assert config["datasets"][0]["overrides"]["data_online_creation_crop_size_A"] == 384


def test_multi_dataset_generator_train_config_matches_b2b_defaults(tmp_path):
    config_path = tmp_path / "multi_dataset_config.json"
    args = make_generator_args(
        name="run_name",
        gpu_ids="1",
        reference_dataroot="/data/reference",
    )

    train_config = build_train_config(args, config_path)

    assert train_config["name"] == "run_name"
    assert train_config["dataroot"] == "/data/reference"
    assert train_config["data_dataset_mode"] == "multi_dataset"
    assert train_config["data_multi_dataset_config"] == str(config_path)
    assert train_config["train_batch_size"] == 8
    assert train_config["train_iter_size"] == 4
    assert train_config["train_n_epochs"] == 6000
    assert train_config["G_netG"] == "vit_vid"
    assert train_config["G_vit_variant"] == "JiT-B/16"
    assert train_config["G_vit_num_classes"] == 3
    assert train_config["f_s_semantic_nclasses"] == 3
    assert train_config["train_optim"] == "muon"
    assert train_config["alg_b2b_mask_as_channel"] is True
    assert train_config["alg_b2b_multi_dataset_class_conditioning"] is False
    assert train_config["alg_b2b_denoise_timesteps"] == [2, 5, 20]
    assert train_config["alg_b2b_perceptual_loss"] == ["LPIPS", "DISTS"]
    assert "data_online_creation_crop_size_A" not in train_config
    assert "data_online_creation_crop_delta_A" not in train_config


def test_multi_dataset_generator_train_config_can_enable_dataset_conditioning(tmp_path):
    config_path = tmp_path / "multi_dataset_config.json"
    args = make_generator_args(
        alg_b2b_multi_dataset_class_conditioning=True,
        multi_dataset_num_datasets=5,
    )

    train_config = build_train_config(args, config_path)

    assert train_config["alg_b2b_multi_dataset_class_conditioning"] is True
    assert train_config["G_vit_num_classes"] == 5
    assert train_config["alg_b2b_mask_as_channel"] is True
    assert train_config["f_s_semantic_nclasses"] == 3


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
    monkeypatch.setattr(
        data_module, "find_dataset_using_name", lambda _: FakeChildDataset
    )
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
    monkeypatch.setattr(
        data_module, "find_dataset_using_name", lambda _: FakeChildDataset
    )
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
        f"vid/frame_{index:03d}.png masks/frame_{index:03d}.png" for index in range(6)
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
