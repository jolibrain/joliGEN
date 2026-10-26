"""Derive isolated ConvRefiner configs and recording-level holdout manifests."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re

DEFAULT_SOURCE = "/data1/beniz/checkpoints/beniz_lvring_ft_32bit_rotrebox_handcond_2_smartbrush_4_new_aug_ft3/train_config.json"


def recording_key(path):
    match = re.fullmatch(r"(.+?)[_-](\d+)", Path(path).stem)
    if not match:
        raise ValueError(f"Cannot identify recording/frame number in {path}")
    return str(Path(path).parent / match.group(1))


def prepare(source_config, output_dir, epoch="320", source_ema=True, seed=42):
    source_path = Path(source_config).resolve()
    config = json.loads(source_path.read_text())
    if (
        config["model_type"] != "b2b"
        or config["data"]["dataset_mode"] != "self_supervised_vid_mask_online"
    ):
        raise ValueError(
            "This recipe generator expects a self_supervised_vid_mask_online B2B run"
        )
    if config["alg"].get("b2b_multi_dataset_class_conditioning", False):
        raise ValueError(
            "Recording splitting must not change dataset-index class conditioning"
        )
    root = Path(config["dataroot"])
    groups = {}
    for line in (root / "trainA" / "paths.txt").read_text().splitlines():
        if not line.strip():
            continue
        image, label = line.split()
        image = str((root / image).resolve())
        label = str((root / label).resolve())
        groups.setdefault(recording_key(image), []).append(f"{image} {label}")
    minimum = 1 + (config["data"]["temporal_number_frames"] - 1) * config["data"].get(
        "temporal_frame_step", 1
    )
    if any(len(rows) < minimum for rows in groups.values()):
        raise ValueError(
            "Every recording must contain at least one complete training window"
        )
    if len(groups) < 2:
        raise ValueError("Need at least two recordings for a disjoint holdout")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    heldout = set(keys[: max(1, math.ceil(len(keys) * 0.1))])
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing recipe directory: {destination}"
        )
    destination.mkdir(parents=True)
    datasets, test_sets = [], []
    split = {"seed": seed, "source_config": str(source_path), "train": [], "test": []}
    for key in sorted(groups):
        name = Path(key).name + "_" + hashlib.sha256(key.encode()).hexdigest()[:8]
        child_root = destination / "data" / name
        phase = "test" if key in heldout else "train"
        manifest = child_root / (phase + "A") / "paths.txt"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("\n".join(groups[key]) + "\n")
        split[phase].append(key)
        if phase == "train":
            datasets.append(
                {
                    "name": name,
                    "dataroot": str(child_root),
                    "dataset_mode": "self_supervised_vid_mask_online",
                    "weight": len(groups[key]),
                }
            )
        else:
            test_sets.append(
                {"id": name, "dataroot": str(child_root), "child_test_name": ""}
            )
    for entry in test_sets:
        # Same child options for every recording; override only its manifest root.
        entry["dataset_name"] = datasets[0]["name"]
    (destination / "datasets.json").write_text(
        json.dumps({"datasets": datasets, "test_sets": test_sets}, indent=2) + "\n"
    )
    (destination / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    config["name"] += "_convrefiner_temporal"
    config.update(with_amp=False, with_tf32=False, with_torch_compile=False)
    config["data"].update(
        dataset_mode="multi_dataset",
        multi_dataset_config=str(destination / "datasets.json"),
    )
    config["alg"].update(
        b2b_convrefiner="temporal",
        b2b_convrefiner_width=64,
        b2b_convrefiner_finetune=True,
        b2b_convrefiner_source_ema=source_ema,
        b2b_convrefiner_gan_weight=0.02,
        b2b_convrefiner_eval_batches=16,
        b2b_lora=False,
    )
    config["train"].update(
        continue_from=str(source_path.parent),
        epoch=str(epoch),
        epoch_count=1,
        n_epochs=30,
        n_epochs_decay=0,
        G_lr=1e-4,
        D_lr=1e-4,
        optim="adamw",
        beta1=0.9,
        beta2=0.95,
        optim_weight_decay=0.0,
        batch_size=6,
        iter_size=32,
        save_epoch_freq=1,
        finetune=False,
        load_iter=0,
        G_ema=True,
    )
    config["train"]["continue"] = False
    (destination / "pilot.json").write_text(json.dumps(config, indent=2) + "\n")
    smoke = copy.deepcopy(config)
    smoke["name"] += "_smoke"
    smoke["train"].update(
        n_epochs=1, batch_size=1, iter_size=1, compute_metrics_test=False
    )
    smoke["alg"]["b2b_convrefiner_eval_batches"] = 1
    smoke["data"]["max_dataset_size"] = 2
    smoke["data"]["num_threads"] = 0
    (destination / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_config", default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epoch", default="320")
    parser.add_argument("--raw_source", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        prepare(
            args.source_config,
            args.output_dir,
            args.epoch,
            not args.raw_source,
            args.seed,
        )
    )


if __name__ == "__main__":
    main()
