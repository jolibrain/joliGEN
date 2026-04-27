#!/usr/bin/env python3
import argparse
import glob
import json
import logging
import math
import random
import re
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import create_dataset
from options.train_options import TrainOptions


LOGGER = logging.getLogger("gen_multi_dataset_b2b_config")


try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


def iter_progress(iterable, **kwargs):
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, **kwargs)


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def sanitize_id(value):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "dataset"


def make_test_id(dataset_name, child_test_name):
    base = sanitize_id(dataset_name)
    suffix = sanitize_id(child_test_name) if child_test_name else ""
    return f"{base}__{suffix}" if suffix else base


def natural_keys(text):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def clean_name(folder_name):
    tokens = [token for token in folder_name.split("_") if token]
    filtered = [
        token for token in tokens if token.lower() not in {"online", "clean"}
    ]
    return "_".join(filtered) if filtered else folder_name


def hdi(values, coverage=0.75):
    values = sorted(values)
    if not values:
        return None, None

    window_size = max(1, math.ceil(coverage * len(values)))
    best_index = 0
    best_width = float("inf")
    for index in range(0, len(values) - window_size + 1):
        width = values[index + window_size - 1] - values[index]
        if width < best_width:
            best_width = width
            best_index = index

    return values[best_index], values[best_index + window_size - 1]


def floor_to_multiple(value, step=16):
    return int(math.floor(value / step) * step)


def compute_bbox_stats(bbox_files, coverage, step, ignore_categories, source_label):
    widths, heights, long_sides = [], [], []
    ignored = set(str(category) for category in ignore_categories)

    for bbox_file in iter_progress(
        bbox_files,
        desc=f"Reading bboxes ({Path(source_label).parent.parent.name})",
        unit="file",
        leave=False,
    ):
        with open(bbox_file, "r") as file:
            for line in file:
                parts = line.split()
                if len(parts) < 5:
                    continue
                if parts[0] in ignored:
                    continue
                xmin, ymin, xmax, ymax = map(int, parts[1:5])
                width = xmax - xmin
                height = ymax - ymin
                widths.append(width)
                heights.append(height)
                long_sides.append(max(width, height))

    if not long_sides:
        raise ValueError(f"No usable bboxes found in: {source_label}")

    hdi_lo, hdi_hi = hdi(long_sides, coverage=coverage)
    max_long = max(long_sides)
    raw_target = 0.696 * hdi_hi + 124.5
    capped_target = min(raw_target, max_long - 1e-6)
    final_target = floor_to_multiple(capped_target, step)
    if final_target <= 0:
        raise ValueError(
            f"bbox-derived crop size is not positive for: {source_label}"
        )

    LOGGER.info(
        "Derived crop size for %s: size=%s, count=%s, hdi%s=[%.2f, %.2f], "
        "max_long=%.2f",
        source_label,
        final_target,
        len(long_sides),
        int(coverage * 100),
        hdi_lo,
        hdi_hi,
        max_long,
    )

    return {
        "count": len(long_sides),
        "max_size": max(max(widths), max(heights)),
        "avg_width": sum(widths) / float(len(widths)),
        "avg_height": sum(heights) / float(len(heights)),
        "hdi_lo": hdi_lo,
        "hdi_hi": hdi_hi,
        "max_long": max_long,
        "raw_target": raw_target,
        "capped_target": capped_target,
        "final_target": final_target,
    }


def collect_bboxes_from_paths_file(paths_file):
    LOGGER.info("Reading dataset paths: %s", paths_file)
    if not paths_file.exists():
        if paths_file.is_symlink():
            raise ValueError(f"paths file is a broken symlink: {paths_file}")
        raise ValueError(f"paths file does not exist: {paths_file}")

    dataset_root = paths_file.parent.parent
    bbox_files = []

    with open(paths_file, "r") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid line {line_number} in {paths_file}: expected "
                    "'<image> <bbox>'"
                )

            bbox_path = Path(parts[1])

            if not bbox_path.is_absolute():
                bbox_path = dataset_root / bbox_path
            bbox_path = bbox_path.resolve()
            if not bbox_path.exists():
                raise ValueError(
                    f"bbox file from {paths_file} line {line_number} does not "
                    f"exist: {bbox_path}"
                )
            bbox_files.append(bbox_path)

    if not bbox_files:
        raise ValueError(f"No bbox entries found in: {paths_file}")

    LOGGER.info("Found %d bbox files for dataset root %s", len(bbox_files), dataset_root)
    return bbox_files


def discover_dataset_roots(args):
    if bool(args.datasets_root) == bool(args.dataset_dirs):
        raise ValueError("Provide exactly one of --datasets-root or --dataset-dirs")

    if args.datasets_root:
        root = Path(args.datasets_root).resolve()
        if not root.is_dir():
            raise ValueError(f"--datasets-root is not a directory: {root}")
        LOGGER.info("Discovering datasets under: %s", root)
        dataset_roots = []
        skipped = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if (child / "trainA").exists():
                dataset_roots.append(child)
            else:
                skipped.append(child)
        if skipped:
            LOGGER.warning(
                "Skipping %d direct child directories without trainA: %s",
                len(skipped),
                ", ".join(path.name for path in skipped),
            )
    else:
        dataset_roots = [Path(path).resolve() for path in args.dataset_dirs]
        missing = [path for path in dataset_roots if not (path / "trainA").exists()]
        if missing:
            raise ValueError(
                "Every --dataset-dirs entry must contain a trainA directory; "
                f"missing for: {missing}"
            )

    if not dataset_roots:
        raise ValueError("No dataset roots with trainA directories were found")

    LOGGER.info("Found %d dataset roots", len(dataset_roots))
    return dataset_roots


def build_dataset_entry(dataroot, args):
    LOGGER.info("Processing dataset root: %s", dataroot)
    paths_file = dataroot / "trainA" / "paths.txt"
    name = clean_name(dataroot.name)
    bbox_files = collect_bboxes_from_paths_file(paths_file)
    if args.size is None:
        stats = compute_bbox_stats(
            bbox_files,
            args.coverage,
            args.step,
            args.ignore_categories,
            paths_file,
        )
        crop_size = stats["final_target"]
    else:
        crop_size = args.size
        LOGGER.info("Using manual crop size for %s: size=%s", dataroot, crop_size)

    entry = {
        "name": name,
        "dataset_mode": "self_supervised_vid_mask_online",
        "dataroot": str(dataroot),
        "weight": args.weight,
        "overrides": {
            "data_online_creation_crop_size_A": crop_size,
            "data_online_creation_crop_delta_A": int(round(crop_size * 0.1)),
        },
    }
    LOGGER.info(
        "Configured dataset '%s': dataroot=%s, crop_size_A=%s, crop_delta_A=%s",
        entry["name"],
        entry["dataroot"],
        entry["overrides"]["data_online_creation_crop_size_A"],
        entry["overrides"]["data_online_creation_crop_delta_A"],
    )
    return entry


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
    if test_sets:
        LOGGER.info(
            "Using %d predefined test sets for '%s'",
            len(test_sets),
            entry["name"],
        )
    return test_sets


def valid_temporal_windows(lines, num_frames, frame_step, num_common_char):
    indexed_lines = sorted(enumerate(lines), key=lambda item: natural_keys(item[1]))
    sorted_indices = [item[0] for item in indexed_lines]
    sorted_lines = [item[1] for item in indexed_lines]
    windows = []

    span = (num_frames - 1) * frame_step
    starts = range(0, len(sorted_lines) - span)
    for start in iter_progress(
        starts,
        desc="Scanning temporal windows",
        unit="window",
        leave=False,
        disable=len(sorted_lines) < 1000,
    ):
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
    LOGGER.info("Generating holdout test set for '%s'", entry["name"])
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
    LOGGER.info(
        "Sampled %d temporal windows (%d frame rows) for '%s'",
        sample_count,
        len(holdout_indices),
        entry["name"],
    )

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
    LOGGER.info(
        "Generated holdout root for '%s': %s (train rows=%d, test rows=%d)",
        entry["name"],
        generated_root,
        len(train_lines),
        len(holdout_indices),
    )
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
    for entry in iter_progress(
        config["datasets"],
        desc="Preparing test sets",
        unit="dataset",
    ):
        entry_test_sets = discover_existing_test_sets(entry)
        if not entry_test_sets:
            entry_test_sets = [generate_holdout_test_set(entry, output_dir, args)]

        for test_set in entry_test_sets:
            if test_set["id"] in seen_ids:
                raise ValueError(f"duplicate multi_dataset test id '{test_set['id']}'")
            seen_ids.add(test_set["id"])
            test_sets.append(test_set)

    config["test_sets"] = test_sets
    LOGGER.info("Configured %d multi-dataset test sets", len(test_sets))
    return config


def build_multi_dataset_config(dataset_roots, output_dir=None, args=None):
    datasets = [
        build_dataset_entry(Path(dataroot), args)
        for dataroot in iter_progress(
            dataset_roots,
            desc="Processing datasets",
            unit="dataset",
        )
    ]
    ids = [sanitize_id(entry["name"]) for entry in datasets]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate dataset names after sanitizing: {ids}")

    config = {"datasets": datasets}
    if output_dir is not None and args is not None:
        config = add_test_sets(config, Path(output_dir), args)
    return config


def build_train_config(args, multi_dataset_config_path):
    train_config = {
        "name": args.name,
        "model_type": "b2b",
        "checkpoints_dir": args.checkpoints_dir,
        "gpu_ids": args.gpu_ids,
        "model_input_nc": 3,
        "model_output_nc": 3,
        "data_dataset_mode": "multi_dataset",
        "data_multi_dataset_config": str(multi_dataset_config_path),
        "dataroot": args.reference_dataroot,
        "data_relative_paths": True,
        "G_netG": "vit_vid",
        "G_vit_variant": "JiT-B/16",
        "G_vit_disable_bottleneck": True,
        "data_load_size": args.data_load_size,
        "data_crop_size": args.data_crop_size,
        "data_temporal_number_frames": args.data_temporal_number_frames,
        "data_temporal_frame_step": args.data_temporal_frame_step,
        "data_online_creation_rand_mask_A": True,
        "data_num_threads": args.data_num_threads,
        "dataaug_flip": "both",
        "dataaug_diff_aug_policy": "color",
        "train_batch_size": args.train_batch_size,
        "train_iter_size": args.train_iter_size,
        "train_n_epochs": args.train_n_epochs,
        "train_n_epochs_decay": args.train_n_epochs_decay,
        "train_save_epoch_freq": args.train_save_epoch_freq,
        "train_G_ema": True,
        "train_G_lr": args.train_G_lr,
        "train_optim": "muon",
        "train_optim_weight_decay": 0.0,
        "train_beta1": 0.9,
        "train_beta2": 0.95,
        "train_compute_metrics_test": True,
        "train_metrics_list": ["PSNR", "FID"],
        "train_metrics_every": args.train_metrics_every,
        "output_print_freq": args.output_print_freq,
        "output_display_freq": args.output_display_freq,
        "with_amp": True,
        "with_tf32": True,
        "with_torch_compile": True,
        "alg_b2b_denoise_timesteps": [2, 5, 20],
        "alg_b2b_cfg_scale": 1.0,
        "alg_b2b_disable_inference_clipping": True,
        "alg_b2b_perceptual_loss": ["LPIPS", "DISTS"],
        "alg_b2b_lambda_perceptual": 0.1,
        "alg_b2b_loss": "pseudo_huber",
        "alg_b2b_loss_masked_region_only": True,
        "alg_b2b_autoregressive": True,
        "alg_b2b_use_gt_prob": 0.1,
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

    for entry in iter_progress(
        multi_config["datasets"],
        desc="Writing previews",
        unit="dataset",
    ):
        LOGGER.info("Generating %d preview samples for '%s'", num_samples, entry["name"])
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
        progress = iter_progress(
            range(num_samples),
            desc=f"Preview samples ({entry['name']})",
            unit="sample",
            leave=False,
        )
        for _ in progress:
            while attempts < num_samples * 20:
                sample = dataset[attempts % len(dataset)]
                attempts += 1
                if sample is None:
                    continue
                samples.append(tensor_grid(sample["A"]))
                masks.append(tensor_grid(sample["B_label_mask"]))
                break

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
        LOGGER.info("Wrote previews for '%s' under %s", entry["name"], dataset_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multi-dataset B2B training config and previews."
    )
    parser.add_argument(
        "--datasets-root",
        default="",
        help="directory containing one subdirectory per dataset root",
    )
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=None,
        help="explicit list of dataset roots when they are not under one directory",
    )
    parser.add_argument("--output-dir", required=True, help="directory for generated files")
    parser.add_argument("--name", default="b2b_multi_dataset")
    parser.add_argument("--checkpoints-dir", default="./checkpoints")
    parser.add_argument("--gpu-ids", default="-1")
    parser.add_argument("--base-train-config", default="")
    parser.add_argument("--coverage", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument(
        "--ignore-categories",
        nargs="*",
        default=["2"],
        help=(
            "bbox categories to ignore while deriving crop size; pass with no "
            "values to ignore none"
        ),
    )
    parser.add_argument("--data-load-size", type=int, default=256)
    parser.add_argument("--data-crop-size", type=int, default=256)
    parser.add_argument("--data-temporal-number-frames", type=int, default=2)
    parser.add_argument("--data-temporal-frame-step", type=int, default=1)
    parser.add_argument("--data-num-threads", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-iter-size", type=int, default=4)
    parser.add_argument("--train-n-epochs", type=int, default=6000)
    parser.add_argument("--train-n-epochs-decay", type=int, default=0)
    parser.add_argument("--train-save-epoch-freq", type=int, default=1000)
    parser.add_argument("--train-G-lr", type=float, default=1e-4)
    parser.add_argument("--train-metrics-every", type=int, default=20000)
    parser.add_argument("--output-print-freq", type=int, default=200)
    parser.add_argument("--output-display-freq", type=int, default=1000)
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=0,
        help="number of preview samples per dataset; 0 disables previews",
    )
    parser.add_argument("--auto-test-samples", type=int, default=32)
    parser.add_argument("--auto-test-seed", type=int, default=1337)
    parser.add_argument("--skip-preview", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    dataset_roots = discover_dataset_roots(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    multi_config = build_multi_dataset_config(dataset_roots, output_dir, args)
    multi_config_path = output_dir / "multi_dataset_config.json"
    train_config_path = output_dir / "train_config.json"

    args.reference_dataroot = multi_config["datasets"][0]["dataroot"]
    train_config = build_train_config(args, multi_config_path)

    with open(multi_config_path, "w") as config_file:
        json.dump(multi_config, config_file, indent=2)
        config_file.write("\n")
    LOGGER.info("Wrote multi-dataset config: %s", multi_config_path)
    with open(train_config_path, "w") as config_file:
        json.dump(train_config, config_file, indent=2)
        config_file.write("\n")
    LOGGER.info("Wrote train config: %s", train_config_path)

    if not args.skip_preview and args.preview_samples > 0:
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
