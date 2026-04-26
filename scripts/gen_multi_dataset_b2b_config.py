#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import random
import re
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import create_dataset
from options.train_options import TrainOptions


OVERRIDE_COLUMNS = {
    "data_online_creation_crop_size_A",
    "data_online_creation_crop_delta_A",
    "data_online_creation_load_size_A",
    "data_online_creation_mask_delta_A",
    "data_online_creation_mask_delta_A_ratio",
    "data_online_creation_mask_random_offset_A",
    "data_online_creation_mask_square_A",
    "data_temporal_num_common_char",
}

FORBIDDEN_OVERRIDE_COLUMNS = {
    "data_load_size",
    "data_crop_size",
    "model_input_nc",
    "model_output_nc",
    "data_temporal_number_frames",
    "data_temporal_frame_step",
    "G_netG",
    "G_vit_num_classes",
    "model_type",
    "alg_b2b_mask_as_channel",
    "alg_diffusion_cond_image_creation",
}


def parse_value(value):
    if value is None or value == "":
        return None
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def read_manifest(path):
    with open(path, "r", newline="") as manifest_file:
        sample = manifest_file.read(4096)
        manifest_file.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(manifest_file, dialect=dialect)
        rows = list(reader)

    if not rows:
        raise ValueError("manifest is empty")

    for column in ("name", "dataroot"):
        if column not in rows[0]:
            raise ValueError(f"manifest is missing required column '{column}'")

    return rows


def sanitize_id(value):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "dataset"


def make_test_id(dataset_name, child_test_name):
    base = sanitize_id(dataset_name)
    suffix = sanitize_id(child_test_name) if child_test_name else ""
    return f"{base}__{suffix}" if suffix else base


def natural_keys(text):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def absolutize_paths_line(line, dataroot):
    parts = line.strip().split()
    if not parts:
        return ""

    absolute_parts = []
    for part in parts:
        path = Path(part)
        if not path.is_absolute():
            path = Path(dataroot) / path
        absolute_parts.append(str(path))
    return " ".join(absolute_parts)


def read_paths_file(paths_file):
    with open(paths_file, "r") as file:
        return [line.strip() for line in file.readlines() if line.strip()]


def discover_existing_test_sets(entry):
    test_sets = []
    for test_dir in sorted(glob.glob(str(Path(entry["dataroot"]) / "testA*"))):
        test_dir_path = Path(test_dir)
        if not (test_dir_path / "paths.txt").is_file():
            continue
        child_test_name = test_dir_path.name[len("testA") :]
        test_sets.append(
            {
                "id": make_test_id(entry["name"], child_test_name),
                "dataset_name": entry["name"],
                "dataroot": entry["dataroot"],
                "child_test_name": child_test_name,
                "generated": False,
            }
        )
    return test_sets


def valid_temporal_windows(lines, num_frames, frame_step, num_common_char):
    indexed_lines = sorted(enumerate(lines), key=lambda item: natural_keys(item[1]))
    sorted_indices = [item[0] for item in indexed_lines]
    sorted_lines = [item[1] for item in indexed_lines]
    windows = []

    span = (num_frames - 1) * frame_step
    for start in range(0, len(sorted_lines) - span):
        selected_positions = [start + frame_idx * frame_step for frame_idx in range(num_frames)]
        selected_lines = [sorted_lines[position] for position in selected_positions]
        selected_paths = [Path(line.split()[0]) for line in selected_lines]

        if len({str(path.parent) for path in selected_paths}) != 1:
            continue

        if num_common_char != -1:
            ref_prefix = selected_paths[0].name[:num_common_char]
            if any(path.name[:num_common_char] != ref_prefix for path in selected_paths):
                continue

        windows.append([sorted_indices[position] for position in selected_positions])

    return windows


def generate_holdout_test_set(entry, output_dir, args):
    dataroot = Path(entry["dataroot"])
    train_paths_file = dataroot / "trainA" / "paths.txt"
    if not train_paths_file.is_file():
        raise ValueError(f"missing train paths file: {train_paths_file}")

    lines = read_paths_file(train_paths_file)
    num_common_char = entry.get("overrides", {}).get(
        "data_temporal_num_common_char", -1
    )
    windows = valid_temporal_windows(
        lines,
        args.data_temporal_number_frames,
        args.data_temporal_frame_step,
        num_common_char,
    )
    if not windows:
        raise ValueError(
            f"dataset '{entry['name']}' has no valid temporal windows to sample"
        )

    name_seed = sum(ord(char) for char in entry["name"])
    rng = random.Random(args.auto_test_seed + name_seed)
    sample_count = min(args.auto_test_samples, len(windows))
    sampled_windows = rng.sample(windows, sample_count)
    holdout_indices = sorted({index for window in sampled_windows for index in window})

    train_indices = [index for index in range(len(lines)) if index not in holdout_indices]
    if not train_indices:
        raise ValueError(
            f"dataset '{entry['name']}' automatic holdout would leave no train samples"
        )
    train_lines = [lines[index] for index in train_indices]
    if not valid_temporal_windows(
        train_lines,
        args.data_temporal_number_frames,
        args.data_temporal_frame_step,
        num_common_char,
    ):
        raise ValueError(
            f"dataset '{entry['name']}' automatic holdout would leave no valid "
            "train temporal windows"
        )

    generated_root = output_dir / "generated_test_sets" / sanitize_id(entry["name"])
    generated_train_dir = generated_root / "trainA"
    generated_test_dir = generated_root / "testA"
    generated_train_dir.mkdir(parents=True, exist_ok=True)
    generated_test_dir.mkdir(parents=True, exist_ok=True)

    with open(generated_train_dir / "paths.txt", "w") as train_file:
        for line in train_lines:
            train_file.write(absolutize_paths_line(line, dataroot) + "\n")

    with open(generated_test_dir / "paths.txt", "w") as test_file:
        for index in holdout_indices:
            test_file.write(absolutize_paths_line(lines[index], dataroot) + "\n")

    entry["dataroot"] = str(generated_root)
    return {
        "id": make_test_id(entry["name"], ""),
        "dataset_name": entry["name"],
        "dataroot": str(generated_root),
        "child_test_name": "",
        "generated": True,
    }


def add_test_sets(config, output_dir, args):
    test_sets = []
    seen_ids = set()
    for entry in config["datasets"]:
        entry_test_sets = discover_existing_test_sets(entry)
        if not entry_test_sets:
            entry_test_sets = [generate_holdout_test_set(entry, output_dir, args)]

        for test_set in entry_test_sets:
            if test_set["id"] in seen_ids:
                raise ValueError(f"duplicate multi_dataset test id '{test_set['id']}'")
            seen_ids.add(test_set["id"])
            test_sets.append(test_set)

    config["test_sets"] = test_sets
    return config


def build_multi_dataset_config(rows, output_dir=None, args=None):
    datasets = []
    for row_index, row in enumerate(rows):
        for key in FORBIDDEN_OVERRIDE_COLUMNS:
            if row.get(key):
                raise ValueError(
                    f"row {row_index + 1} cannot override shape/model option '{key}'"
                )

        overrides = {}
        for key in OVERRIDE_COLUMNS:
            value = parse_value(row.get(key))
            if value is not None:
                overrides[key] = value

        entry = {
            "name": row["name"],
            "dataset_mode": row.get("dataset_mode") or "self_supervised_vid_mask_online",
            "dataroot": row["dataroot"],
            "weight": float(row["weight"]) if row.get("weight") else 1.0,
            "overrides": overrides,
        }
        datasets.append(entry)

    config = {"datasets": datasets}
    if output_dir is not None and args is not None:
        config = add_test_sets(config, Path(output_dir), args)
    return config


def build_train_config(args, multi_dataset_config_path):
    train_config = {
        "name": args.name,
        "model_type": "b2b",
        "data_dataset_mode": "multi_dataset",
        "data_multi_dataset_config": str(multi_dataset_config_path),
        "dataroot": args.reference_dataroot,
        "checkpoints_dir": args.checkpoints_dir,
        "gpu_ids": args.gpu_ids,
        "G_netG": "vit_vid",
        "data_load_size": args.data_load_size,
        "data_crop_size": args.data_crop_size,
        "data_temporal_number_frames": args.data_temporal_number_frames,
        "data_temporal_frame_step": args.data_temporal_frame_step,
        "data_relative_paths": True,
        "data_online_creation_rand_mask_A": True,
        "dataaug_no_rotate": True,
        "train_batch_size": args.train_batch_size,
        "train_n_epochs": args.train_n_epochs,
        "train_n_epochs_decay": args.train_n_epochs_decay,
        "train_G_ema": True,
        "train_export_jit": False,
        "alg_diffusion_cond_image_creation": "y_t",
        "alg_b2b_denoise_timesteps": [1],
    }
    if args.base_train_config:
        with open(args.base_train_config, "r") as config_file:
            base_config = json.load(config_file)
        base_config.update(train_config)
        train_config = base_config
    return train_config


def tensor_grid(tensor):
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 4:
        tensor = tensor
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"unsupported preview tensor shape {tuple(tensor.shape)}")

    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0
    return tensor.clamp(0, 1)


def write_previews(train_config, multi_config, preview_dir, num_samples):
    preview_dir.mkdir(parents=True, exist_ok=True)

    for entry in multi_config["datasets"]:
        child_config = dict(train_config)
        child_config["data_dataset_mode"] = entry["dataset_mode"]
        child_config["dataroot"] = entry["dataroot"]
        for key, value in entry.get("overrides", {}).items():
            child_config[key] = value

        opt = TrainOptions().parse_json(child_config, save_config=False, set_device=False)
        dataset = create_dataset(opt, "train")
        samples = []
        masks = []
        attempts = 0
        while len(samples) < num_samples and attempts < num_samples * 20:
            sample = dataset[attempts % len(dataset)]
            attempts += 1
            if sample is None:
                continue
            samples.append(tensor_grid(sample["A"]))
            masks.append(tensor_grid(sample["B_label_mask"]))

        if len(samples) != num_samples:
            raise RuntimeError(
                f"dataset '{entry['name']}' produced {len(samples)} valid preview "
                f"samples out of {num_samples}"
            )

        dataset_dir = preview_dir / entry["name"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        save_image(
            torch.cat(samples, dim=0),
            dataset_dir / "A.png",
            nrow=opt.data_temporal_number_frames,
        )
        save_image(
            torch.cat(masks, dim=0),
            dataset_dir / "B_label_mask.png",
            nrow=opt.data_temporal_number_frames,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multi-dataset B2B training config and previews."
    )
    parser.add_argument("--manifest", required=True, help="CSV/TSV dataset manifest")
    parser.add_argument("--output-dir", required=True, help="directory for generated files")
    parser.add_argument("--name", default="b2b_multi_dataset")
    parser.add_argument("--checkpoints-dir", default="./checkpoints")
    parser.add_argument("--gpu-ids", default="-1")
    parser.add_argument("--base-train-config", default="")
    parser.add_argument("--data-load-size", type=int, default=256)
    parser.add_argument("--data-crop-size", type=int, default=256)
    parser.add_argument("--data-temporal-number-frames", type=int, default=2)
    parser.add_argument("--data-temporal-frame-step", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--train-n-epochs", type=int, default=1)
    parser.add_argument("--train-n-epochs-decay", type=int, default=0)
    parser.add_argument("--preview-samples", type=int, default=4)
    parser.add_argument("--auto-test-samples", type=int, default=32)
    parser.add_argument("--auto-test-seed", type=int, default=1337)
    parser.add_argument("--skip-preview", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_manifest(args.manifest)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    multi_config = build_multi_dataset_config(rows, output_dir, args)
    multi_config_path = output_dir / "multi_dataset_config.json"
    train_config_path = output_dir / "train_config.json"

    args.reference_dataroot = multi_config["datasets"][0]["dataroot"]
    train_config = build_train_config(args, multi_config_path)

    with open(multi_config_path, "w") as config_file:
        json.dump(multi_config, config_file, indent=2)
        config_file.write("\n")
    with open(train_config_path, "w") as config_file:
        json.dump(train_config, config_file, indent=2)
        config_file.write("\n")

    if not args.skip_preview:
        write_previews(
            train_config,
            multi_config,
            output_dir / "previews",
            args.preview_samples,
        )

    print(f"wrote {multi_config_path}")
    print(f"wrote {train_config_path}")


if __name__ == "__main__":
    main()
