#!/usr/bin/env python3
import argparse
import csv
import json
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


def build_multi_dataset_config(rows):
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

    return {"datasets": datasets}


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
    parser.add_argument("--skip-preview", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_manifest(args.manifest)
    multi_config = build_multi_dataset_config(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
