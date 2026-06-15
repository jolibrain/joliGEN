#!/usr/bin/env python3
import argparse
import hashlib
import glob
import json
import logging
import math
import random
import re
import shutil
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data import create_dataset
from options.train_options import TrainOptions

LOGGER = logging.getLogger("gen_multi_dataset_b2b_config")
RESUME_SCHEMA_VERSION = 1
VIDEO_CHILD_DATASET_MODE = "self_supervised_vid_mask_online"
NON_VIDEO_CHILD_DATASET_MODES = {
    "self_supervised_labeled_mask_online",
    "self_supervised_labeled_mask_cls_online",
}
SUPPORTED_CHILD_DATASET_MODES = {
    VIDEO_CHILD_DATASET_MODE,
    *NON_VIDEO_CHILD_DATASET_MODES,
}


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


def is_video_child_dataset_mode(dataset_mode):
    return dataset_mode == VIDEO_CHILD_DATASET_MODE


def child_dataset_mode(args):
    return getattr(args, "child_dataset_mode", VIDEO_CHILD_DATASET_MODE)


def train_netG(args):
    explicit_netG = getattr(args, "G_netG", None)
    if explicit_netG:
        return explicit_netG
    if is_video_child_dataset_mode(child_dataset_mode(args)):
        return "vit_vid"
    return "vit"


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
    filtered = [token for token in tokens if token.lower() not in {"online", "clean"}]
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


def json_fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def path_metadata(path):
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as file:
        file.write(text)
    tmp_path.replace(path)


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


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
        raise ValueError(f"bbox-derived crop size is not positive for: {source_label}")

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

    LOGGER.info(
        "Found %d bbox files for dataset root %s", len(bbox_files), dataset_root
    )
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
                child_dataset_roots = [
                    grandchild
                    for grandchild in sorted(child.iterdir())
                    if grandchild.is_dir() and (grandchild / "trainA").exists()
                ]
                if child_dataset_roots:
                    dataset_roots.extend(child_dataset_roots)
                else:
                    skipped.append(child)
        if skipped:
            LOGGER.warning(
                "Skipping %d direct child directories without trainA or nested datasets: %s",
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


def dataset_entry_name(dataroot, args):
    datasets_root = getattr(args, "datasets_root", "")
    if datasets_root:
        try:
            relative_parts = (
                dataroot.resolve().relative_to(Path(datasets_root).resolve()).parts
            )
        except ValueError:
            relative_parts = ()

        if len(relative_parts) == 1:
            return clean_name(relative_parts[0])
        if len(relative_parts) == 2:
            return "_".join(clean_name(part) for part in relative_parts)

    return clean_name(dataroot.name)


def build_dataset_entry(dataroot, args):
    LOGGER.info("Processing dataset root: %s", dataroot)
    paths_file = dataroot / "trainA" / "paths.txt"
    name = dataset_entry_name(dataroot, args)
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
        "dataset_mode": child_dataset_mode(args),
        "dataroot": str(dataroot),
        "weight": args.weight,
        "overrides": {
            "data_online_creation_crop_size_A": crop_size,
            "data_online_creation_crop_delta_A": int(
                round(crop_size * args.crop_delta_ratio)
            ),
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


def existing_test_paths_metadata(dataroot):
    metadata = []
    for test_dir in sorted(glob.glob(str(Path(dataroot) / "testA*"))):
        paths_file = Path(test_dir) / "paths.txt"
        metadata.append(path_metadata(paths_file))
    return metadata


def dataset_resume_fingerprint(dataroot, name, args):
    paths_file = Path(dataroot) / "trainA" / "paths.txt"
    return json_fingerprint(
        {
            "schema_version": RESUME_SCHEMA_VERSION,
            "name": name,
            "dataroot": str(Path(dataroot).resolve()),
            "train_paths": path_metadata(paths_file),
            "test_paths": existing_test_paths_metadata(dataroot),
            "entry_args": {
                "child_dataset_mode": child_dataset_mode(args),
                "coverage": args.coverage,
                "step": args.step,
                "size": args.size,
                "weight": args.weight,
                "crop_delta_ratio": args.crop_delta_ratio,
                "ignore_categories": list(args.ignore_categories),
            },
            "holdout_args": {
                "data_temporal_number_frames": args.data_temporal_number_frames,
                "data_temporal_frame_step": args.data_temporal_frame_step,
                "auto_test_samples": args.auto_test_samples,
                "auto_test_seed": args.auto_test_seed,
            },
        }
    )


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
        selected_positions = [
            start + frame_idx * frame_step for frame_idx in range(num_frames)
        ]
        selected_lines = [sorted_lines[position] for position in selected_positions]
        selected_paths = [Path(line.split()[0]) for line in selected_lines]

        if len({str(path.parent) for path in selected_paths}) != 1:
            continue

        if num_common_char != -1:
            ref_prefix = selected_paths[0].name[:num_common_char]
            if any(
                path.name[:num_common_char] != ref_prefix for path in selected_paths
            ):
                continue

        windows.append([sorted_indices[position] for position in selected_positions])

    return windows


def train_has_temporal_window_after_holdout(windows, holdout_indices):
    return any(
        all(index not in holdout_indices for index in window) for window in windows
    )


def sample_holdout_windows_preserving_train(windows, entry_name, args):
    name_seed = sum(ord(char) for char in entry_name)
    rng = random.Random(args.auto_test_seed + name_seed)
    sample_count = min(args.auto_test_samples, len(windows))
    shuffled_windows = list(windows)
    rng.shuffle(shuffled_windows)

    sampled_windows = []
    holdout_indices = set()
    for window in shuffled_windows:
        if len(sampled_windows) >= sample_count:
            break

        candidate_holdout_indices = holdout_indices.union(window)
        if not train_has_temporal_window_after_holdout(
            windows, candidate_holdout_indices
        ):
            continue

        sampled_windows.append(window)
        holdout_indices = candidate_holdout_indices

    if not sampled_windows:
        raise ValueError(
            f"dataset '{entry_name}' automatic holdout cannot preserve a valid "
            "train temporal window"
        )

    if len(sampled_windows) < sample_count:
        LOGGER.warning(
            "Reduced automatic holdout for '%s' from %d to %d temporal windows "
            "to preserve train samples",
            entry_name,
            sample_count,
            len(sampled_windows),
        )

    return sampled_windows, sorted(holdout_indices)


def sample_holdout_rows_preserving_train(lines, entry_name, args):
    name_seed = sum(ord(char) for char in entry_name)
    rng = random.Random(args.auto_test_seed + name_seed)
    sample_count = min(args.auto_test_samples, len(lines))
    if sample_count <= 0:
        raise ValueError(f"dataset '{entry_name}' has no rows to sample")
    if sample_count >= len(lines):
        if len(lines) == 1:
            raise ValueError(
                f"dataset '{entry_name}' automatic holdout would leave no train samples"
            )
        sample_count = len(lines) - 1

    indices = list(range(len(lines)))
    rng.shuffle(indices)
    return sorted(indices[:sample_count])


def generate_holdout_test_set(entry, output_dir, args):
    dataroot = Path(entry["dataroot"])
    LOGGER.info("Generating holdout test set for '%s'", entry["name"])
    train_paths_file = dataroot / "trainA" / "paths.txt"
    if not train_paths_file.is_file():
        raise ValueError(f"missing train paths file: {train_paths_file}")

    lines = read_paths_file(train_paths_file)
    if is_video_child_dataset_mode(entry["dataset_mode"]):
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

        sampled_windows, holdout_indices = sample_holdout_windows_preserving_train(
            windows, entry["name"], args
        )
        LOGGER.info(
            "Sampled %d temporal windows (%d frame rows) for '%s'",
            len(sampled_windows),
            len(holdout_indices),
            entry["name"],
        )
    else:
        num_common_char = -1
        holdout_indices = sample_holdout_rows_preserving_train(
            lines, entry["name"], args
        )
        LOGGER.info(
            "Sampled %d image rows for '%s'",
            len(holdout_indices),
            entry["name"],
        )

    train_indices = [
        index for index in range(len(lines)) if index not in holdout_indices
    ]
    if not train_indices:
        raise ValueError(
            f"dataset '{entry['name']}' automatic holdout would leave no train samples"
        )
    train_lines = [lines[index] for index in train_indices]
    if is_video_child_dataset_mode(
        entry["dataset_mode"]
    ) and not valid_temporal_windows(
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

    train_text = "".join(
        absolutize_paths_line(line, dataroot) + "\n" for line in train_lines
    )
    test_text = "".join(
        absolutize_paths_line(lines[index], dataroot) + "\n"
        for index in holdout_indices
    )
    atomic_write_text(generated_train_dir / "paths.txt", train_text)
    atomic_write_text(generated_test_dir / "paths.txt", test_text)

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


def should_skip_auto_test_set(entry, args):
    if getattr(args, "no_auto_test_holdout", False):
        LOGGER.warning(
            "Skipping automatic test set for '%s': --no-auto-test-holdout is set",
            entry["name"],
        )
        return True

    min_images = getattr(args, "auto_test_min_images", 0)
    if min_images <= 0:
        return False

    paths_file = Path(entry["dataroot"]) / "trainA" / "paths.txt"
    lines = read_paths_file(paths_file)
    if len(lines) >= min_images:
        return False

    LOGGER.warning(
        "Skipping automatic test set for '%s': train rows=%d < "
        "--auto-test-min-images=%d",
        entry["name"],
        len(lines),
        min_images,
    )
    return True


def add_test_sets(config, output_dir, args):
    test_sets = []
    seen_ids = set()
    for entry in iter_progress(
        config["datasets"],
        desc="Preparing test sets",
        unit="dataset",
    ):
        entry_test_sets = discover_existing_test_sets(entry)
        if not entry_test_sets and not should_skip_auto_test_set(entry, args):
            entry_test_sets = [generate_holdout_test_set(entry, output_dir, args)]

        for test_set in entry_test_sets:
            if test_set["id"] in seen_ids:
                raise ValueError(f"duplicate multi_dataset test id '{test_set['id']}'")
            seen_ids.add(test_set["id"])
            test_sets.append(test_set)

    config["test_sets"] = test_sets
    LOGGER.info("Configured %d multi-dataset test sets", len(test_sets))
    return config


def dataset_cache_path(output_dir, dataset_name):
    return (
        Path(output_dir) / "resume" / "datasets" / f"{sanitize_id(dataset_name)}.json"
    )


def has_valid_temporal_paths(paths_file, args, num_common_char=-1):
    if not Path(paths_file).is_file():
        return False
    lines = read_paths_file(paths_file)
    return bool(
        valid_temporal_windows(
            lines,
            args.data_temporal_number_frames,
            args.data_temporal_frame_step,
            num_common_char,
        )
    )


def has_valid_image_paths(paths_file):
    return Path(paths_file).is_file() and bool(read_paths_file(paths_file))


def validate_cached_dataset_artifacts(cache, args):
    entry = cache.get("entry")
    test_sets = cache.get("test_sets")
    if not isinstance(entry, dict) or not isinstance(test_sets, list):
        return False

    source_dataroot = Path(cache.get("source_dataroot", ""))
    if not (source_dataroot / "trainA" / "paths.txt").is_file():
        return False

    num_common_char = entry.get("overrides", {}).get(
        "data_temporal_num_common_char", -1
    )
    is_video_mode = is_video_child_dataset_mode(entry.get("dataset_mode", ""))
    for test_set in test_sets:
        paths_file = Path(test_set.get("dataroot", "")) / "testA" / "paths.txt"
        if test_set.get("generated"):
            train_paths_file = Path(entry.get("dataroot", "")) / "trainA" / "paths.txt"
            if is_video_mode:
                if not has_valid_temporal_paths(
                    train_paths_file, args, num_common_char
                ):
                    return False
                if not has_valid_temporal_paths(paths_file, args, num_common_char):
                    return False
            else:
                if not has_valid_image_paths(train_paths_file):
                    return False
                if not has_valid_image_paths(paths_file):
                    return False
        else:
            child_test_name = test_set.get("child_test_name", "")
            paths_file = (
                Path(test_set.get("dataroot", ""))
                / f"testA{child_test_name}"
                / "paths.txt"
            )
            if not paths_file.is_file():
                return False

    return True


def resume_cache_without_generated_holdouts(cache):
    entry = dict(cache["entry"])
    entry["dataroot"] = cache["source_dataroot"]
    test_sets = [
        test_set
        for test_set in cache.get("test_sets", [])
        if not test_set.get("generated")
    ]
    return entry, test_sets


def load_dataset_cache(cache_path, dataroot, name, fingerprint, args):
    if not getattr(args, "resume", False) or not cache_path.is_file():
        return None
    try:
        with open(cache_path, "r") as cache_file:
            cache = json.load(cache_file)
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Ignoring invalid resume cache %s: %s", cache_path, error)
        return None

    if cache.get("schema_version") != RESUME_SCHEMA_VERSION:
        return None
    if cache.get("fingerprint") != fingerprint:
        return None
    if cache.get("name") != name:
        return None
    if cache.get("source_dataroot") != str(Path(dataroot).resolve()):
        return None
    if getattr(args, "no_auto_test_holdout", False) and any(
        test_set.get("generated") for test_set in cache.get("test_sets", [])
    ):
        if not (
            Path(cache.get("source_dataroot", "")) / "trainA" / "paths.txt"
        ).is_file():
            return None
        entry, test_sets = resume_cache_without_generated_holdouts(cache)
        LOGGER.info(
            "Reusing cached dataset '%s' without generated holdout test sets",
            name,
        )
        return {"entry": entry, "test_sets": test_sets}
    if not validate_cached_dataset_artifacts(cache, args):
        LOGGER.warning("Ignoring incomplete resume cache for '%s'", name)
        return None

    LOGGER.info("Reusing completed dataset '%s' from resume cache", name)
    return cache


def build_dataset_with_test_sets(dataroot, output_dir, args):
    entry = build_dataset_entry(Path(dataroot), args)
    entry_test_sets = discover_existing_test_sets(entry)
    if not entry_test_sets and not should_skip_auto_test_set(entry, args):
        entry_test_sets = [generate_holdout_test_set(entry, output_dir, args)]
    return entry, entry_test_sets


def build_or_resume_dataset(dataroot, output_dir, args):
    dataroot = Path(dataroot)
    name = dataset_entry_name(dataroot, args)
    fingerprint = dataset_resume_fingerprint(dataroot, name, args)
    cache_path = dataset_cache_path(output_dir, name)
    cache = load_dataset_cache(cache_path, dataroot, name, fingerprint, args)
    if cache is not None:
        return cache["entry"], cache["test_sets"]

    entry, test_sets = build_dataset_with_test_sets(dataroot, output_dir, args)
    cache_payload = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "name": name,
        "source_dataroot": str(dataroot.resolve()),
        "entry": entry,
        "test_sets": test_sets,
    }
    atomic_write_json(cache_path, cache_payload)
    return entry, test_sets


def write_resume_progress_config(output_dir, datasets, test_sets, total_count):
    progress_config = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "complete": len(datasets) == total_count,
        "completed_count": len(datasets),
        "total_count": total_count,
        "datasets": datasets,
        "test_sets": test_sets,
    }
    atomic_write_json(
        Path(output_dir) / "resume" / "multi_dataset_config.progress.json",
        progress_config,
    )


def validate_unique_test_set_ids(test_sets):
    seen_ids = set()
    for test_set in test_sets:
        if test_set["id"] in seen_ids:
            raise ValueError(f"duplicate multi_dataset test id '{test_set['id']}'")
        seen_ids.add(test_set["id"])


def build_multi_dataset_config(dataset_roots, output_dir=None, args=None):
    names = [dataset_entry_name(Path(dataroot), args) for dataroot in dataset_roots]
    ids = [sanitize_id(name) for name in names]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate dataset names after sanitizing: {ids}")

    datasets = []
    test_sets = []
    for dataroot in iter_progress(
        dataset_roots,
        desc="Processing datasets",
        unit="dataset",
    ):
        if output_dir is None or args is None:
            entry = build_dataset_entry(Path(dataroot), args)
            datasets.append(entry)
            continue

        entry, entry_test_sets = build_or_resume_dataset(dataroot, output_dir, args)
        datasets.append(entry)
        test_sets.extend(entry_test_sets)
        validate_unique_test_set_ids(test_sets)
        write_resume_progress_config(
            output_dir, datasets, test_sets, len(dataset_roots)
        )

    config = {"datasets": datasets}
    if output_dir is not None and args is not None:
        validate_unique_test_set_ids(test_sets)
        config["test_sets"] = test_sets
        LOGGER.info("Configured %d multi-dataset test sets", len(test_sets))
    return config


def build_train_config(args, multi_dataset_config_path):
    netG = train_netG(args)
    emit_temporal_options = (
        is_video_child_dataset_mode(child_dataset_mode(args)) or netG == "vit_vid"
    )
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
        "G_netG": netG,
        "G_vit_variant": "JiT-B/16",
        "G_vit_num_classes": (
            int(args.multi_dataset_num_datasets)
            if args.alg_b2b_multi_dataset_class_conditioning
            else 3
        ),
        "G_vit_disable_bottleneck": True,
        "f_s_semantic_nclasses": 3,
        "data_load_size": args.data_load_size,
        "data_crop_size": args.data_crop_size,
        **(
            {
                "data_online_creation_load_size_A": args.reference_frame_size,
                "data_online_creation_load_size_keep_ratio_A": True,
            }
            if args.reference_frame_size is not None and args.keep_ratio_load_size
            else {}
        ),
        "data_online_creation_rand_mask_A": True,
        "data_num_threads": args.data_num_threads,
        "dataaug_flip": "both",
        "dataaug_no_rotate": True,
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
        "alg_b2b_mask_as_channel": True,
        "alg_b2b_multi_dataset_class_conditioning": (
            args.alg_b2b_multi_dataset_class_conditioning
        ),
        "alg_b2b_denoise_timesteps": [2, 5, 20],
        "alg_b2b_timestep_uniform_mix_prob": 0.1,
        "alg_b2b_cfg_scale": 1.0,
        "alg_b2b_disable_inference_clipping": True,
        "alg_b2b_perceptual_loss": ["LPIPS", "DISTS"],
        "alg_b2b_lambda_perceptual": 0.1,
        "alg_b2b_loss": "pseudo_huber",
        "alg_b2b_loss_masked_region_only": True,
        "alg_b2b_autoregressive": True,
        "alg_b2b_use_gt_prob": 0.1,
    }
    if emit_temporal_options:
        train_config["data_temporal_number_frames"] = args.data_temporal_number_frames
        train_config["data_temporal_frame_step"] = args.data_temporal_frame_step
    if args.base_train_config:
        with open(args.base_train_config, "r") as config_file:
            base_config = json.load(config_file)
        base_config.update(train_config)
        train_config = base_config
    if not emit_temporal_options:
        train_config.pop("data_temporal_number_frames", None)
        train_config.pop("data_temporal_frame_step", None)
    fixed_mask_size_A = getattr(args, "data_online_creation_mask_fixed_size_A", -1)
    if fixed_mask_size_A > 0:
        train_config["data_online_creation_mask_fixed_size_A"] = fixed_mask_size_A
        train_config["data_online_creation_mask_min_unmasked_border_A"] = (
            args.data_online_creation_mask_min_unmasked_border_A
        )
    if getattr(args, "data_online_creation_mask_broaden_rect_aug_A", False):
        train_config["data_online_creation_mask_broaden_rect_aug_A"] = True
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


def mask_overlay_grid(image_tensor, mask_tensor):
    image = tensor_grid(image_tensor)
    mask = tensor_grid(mask_tensor)[:, :1]
    color = torch.zeros_like(image)
    color[:, 0:1] = 1.0
    return torch.where(mask > 0, image * 0.45 + color * 0.55, image).clamp(0, 1)


def sample_source_stem(sample, fallback_index):
    paths = sample.get("A_img_paths") or sample.get("B_img_paths") or ""
    if isinstance(paths, (list, tuple)):
        paths = paths[0] if paths else ""
    stem = Path(str(paths)).stem if paths else f"sample_{fallback_index:04d}"
    return sanitize_id(stem) or f"sample_{fallback_index:04d}"


def save_named_preview_frames(dataset_dir, sample, sample_index):
    source_stem = sample_source_stem(sample, sample_index)
    preview_tensors = {
        "A": tensor_grid(sample["A"]),
        "B": tensor_grid(sample["B"]),
        "B_label_mask": tensor_grid(sample["B_label_mask"]),
        "B_mask_overlay": mask_overlay_grid(sample["B"], sample["B_label_mask"]),
    }
    named_dir = dataset_dir / "by_image"
    named_dir.mkdir(parents=True, exist_ok=True)

    manifest_entry = {
        "source_path": str(sample.get("A_img_paths", "")),
        "files": [],
    }
    for kind, tensor in preview_tensors.items():
        for frame_index, frame in enumerate(tensor):
            filename = (
                f"{source_stem}_sample{sample_index:04d}_"
                f"frame{frame_index:02d}_{kind}.png"
            )
            save_image(frame, named_dir / filename)
            manifest_entry["files"].append(str(Path("by_image") / filename))
    return manifest_entry


def preview_resume_fingerprint(train_config, entry, num_samples):
    return json_fingerprint(
        {
            "schema_version": RESUME_SCHEMA_VERSION,
            "entry": entry,
            "num_samples": num_samples,
            "train_config": {
                "data_temporal_number_frames": train_config.get(
                    "data_temporal_number_frames"
                ),
                "data_temporal_frame_step": train_config.get(
                    "data_temporal_frame_step"
                ),
                "data_load_size": train_config.get("data_load_size"),
                "data_crop_size": train_config.get("data_crop_size"),
            },
        }
    )


def preview_complete(dataset_dir, fingerprint, num_samples):
    dataset_dir = Path(dataset_dir)
    try:
        with open(dataset_dir / "preview_resume.json", "r") as cache_file:
            cache = json.load(cache_file)
        with open(dataset_dir / "preview_manifest.json", "r") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError):
        return False

    if cache.get("schema_version") != RESUME_SCHEMA_VERSION:
        return False
    if cache.get("fingerprint") != fingerprint:
        return False
    if not isinstance(manifest, list) or len(manifest) != num_samples:
        return False

    required_files = ["A.png", "B.png", "B_label_mask.png", "B_mask_overlay.png"]
    if any(not (dataset_dir / filename).is_file() for filename in required_files):
        return False

    for manifest_entry in manifest:
        for filename in manifest_entry.get("files", []):
            if not (dataset_dir / filename).is_file():
                return False
    return True


def write_preview_for_entry(dataset_dir, train_config, entry, num_samples, fingerprint):
    LOGGER.info("Generating %d preview samples for '%s'", num_samples, entry["name"])
    child_config = dict(train_config)
    child_config["data_dataset_mode"] = entry["dataset_mode"]
    child_config["dataroot"] = entry["dataroot"]
    child_config["dataaug_flip"] = "none"
    child_config["dataaug_no_rotate"] = True
    child_config["dataaug_affine"] = 0.0
    for key, value in entry.get("overrides", {}).items():
        child_config[key] = value

    opt = TrainOptions().parse_json(child_config, save_config=False, set_device=False)
    dataset = create_dataset(opt, "train")
    samples = []
    targets = []
    masks = []
    overlays = []
    manifest = []
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
            targets.append(tensor_grid(sample["B"]))
            masks.append(tensor_grid(sample["B_label_mask"]))
            overlays.append(mask_overlay_grid(sample["B"], sample["B_label_mask"]))
            manifest.append(
                save_named_preview_frames(dataset_dir, sample, len(samples) - 1)
            )
            break

    while len(samples) < num_samples and attempts < num_samples * 20:
        sample = dataset[attempts % len(dataset)]
        attempts += 1
        if sample is None:
            continue
        samples.append(tensor_grid(sample["A"]))
        targets.append(tensor_grid(sample["B"]))
        masks.append(tensor_grid(sample["B_label_mask"]))
        overlays.append(mask_overlay_grid(sample["B"], sample["B_label_mask"]))
        manifest.append(
            save_named_preview_frames(dataset_dir, sample, len(samples) - 1)
        )

    if len(samples) != num_samples:
        raise RuntimeError(
            f"dataset '{entry['name']}' produced {len(samples)} valid preview "
            f"samples out of {num_samples}"
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_image(
        torch.cat(samples, dim=0),
        dataset_dir / "A.png",
        nrow=opt.data_temporal_number_frames,
    )
    save_image(
        torch.cat(targets, dim=0),
        dataset_dir / "B.png",
        nrow=opt.data_temporal_number_frames,
    )
    save_image(
        torch.cat(masks, dim=0),
        dataset_dir / "B_label_mask.png",
        nrow=opt.data_temporal_number_frames,
    )
    save_image(
        torch.cat(overlays, dim=0),
        dataset_dir / "B_mask_overlay.png",
        nrow=opt.data_temporal_number_frames,
    )
    atomic_write_json(dataset_dir / "preview_manifest.json", manifest)
    atomic_write_json(
        dataset_dir / "preview_resume.json",
        {"schema_version": RESUME_SCHEMA_VERSION, "fingerprint": fingerprint},
    )


def write_previews(train_config, multi_config, preview_dir, num_samples, resume=False):
    preview_dir.mkdir(parents=True, exist_ok=True)

    for entry in iter_progress(
        multi_config["datasets"],
        desc="Writing previews",
        unit="dataset",
    ):
        dataset_dir = preview_dir / entry["name"]
        fingerprint = preview_resume_fingerprint(train_config, entry, num_samples)
        if resume and preview_complete(dataset_dir, fingerprint, num_samples):
            LOGGER.info("Reusing completed previews for '%s'", entry["name"])
            continue

        tmp_dataset_dir = preview_dir / f".{sanitize_id(entry['name'])}.tmp"
        if tmp_dataset_dir.exists():
            shutil.rmtree(tmp_dataset_dir)
        write_preview_for_entry(
            tmp_dataset_dir, train_config, entry, num_samples, fingerprint
        )
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        tmp_dataset_dir.replace(dataset_dir)
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
    parser.add_argument(
        "--output-dir", required=True, help="directory for generated files"
    )
    parser.add_argument("--name", default="b2b_multi_dataset")
    parser.add_argument("--checkpoints-dir", default="./checkpoints")
    parser.add_argument("--gpu-ids", default="-1")
    parser.add_argument("--base-train-config", default="")
    parser.add_argument("--coverage", type=float, default=0.75)
    parser.add_argument("--step", type=int, default=16)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument(
        "--crop-delta-ratio",
        type=float,
        default=0.1,
        help=(
            "per-dataset data_online_creation_crop_delta_A ratio relative to "
            "the derived or manual crop size"
        ),
    )
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
    parser.add_argument(
        "--child-dataset-mode",
        choices=sorted(SUPPORTED_CHILD_DATASET_MODES),
        default=VIDEO_CHILD_DATASET_MODE,
        help="dataset mode used for every generated multi_dataset child entry",
    )
    parser.add_argument(
        "--G-netG",
        choices=["vit_vid", "vit"],
        default=None,
        help=(
            "network type for train_config.json; defaults to vit_vid for video "
            "children and vit for non-video children"
        ),
    )
    parser.add_argument(
        "--reference-frame-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="reference frame size for online loading; the largest side is used as target largest image side with --keep-ratio-load-size",
    )
    parser.add_argument(
        "--keep-ratio-load-size",
        action="store_true",
        help="emit aspect-preserving online load-size settings in train_config.json",
    )
    parser.add_argument(
        "--data-online-creation-mask-fixed-size-A", type=int, default=-1
    )
    parser.add_argument(
        "--data-online-creation-mask-min-unmasked-border-A", type=int, default=4
    )
    parser.add_argument(
        "--data-online-creation-mask-broaden-rect-aug-A",
        action="store_true",
        help="emit rectangular bbox broadening augmentation for domain A online masks",
    )
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
        "--alg-b2b-multi-dataset-class-conditioning",
        action="store_true",
        help=(
            "condition B2B ViT class tokens on the multi-dataset dataset index "
            "instead of object labels"
        ),
    )
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=0,
        help="number of preview samples per dataset; 0 disables previews",
    )
    parser.add_argument("--auto-test-samples", type=int, default=32)
    parser.add_argument("--auto-test-seed", type=int, default=1337)
    parser.add_argument(
        "--auto-test-min-images",
        type=int,
        default=1000,
        help=(
            "minimum train rows required to generate an automatic test holdout; "
            "datasets below this value and without predefined testA* sets get no "
            "generated test set"
        ),
    )
    parser.add_argument(
        "--no-auto-test-holdout",
        action="store_true",
        help=(
            "do not generate automatic holdout test sets for datasets without "
            "predefined testA* directories"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed per-dataset outputs from --output-dir when valid",
    )
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
    args.multi_dataset_num_datasets = len(multi_config["datasets"])
    train_config = build_train_config(args, multi_config_path)

    atomic_write_json(multi_config_path, multi_config)
    LOGGER.info("Wrote multi-dataset config: %s", multi_config_path)
    atomic_write_json(train_config_path, train_config)
    LOGGER.info("Wrote train config: %s", train_config_path)

    if not args.skip_preview and args.preview_samples > 0:
        write_previews(
            train_config,
            multi_config,
            output_dir / "previews",
            args.preview_samples,
            resume=args.resume,
        )

    print(f"wrote {multi_config_path}")
    print(f"wrote {train_config_path}")


if __name__ == "__main__":
    main()
