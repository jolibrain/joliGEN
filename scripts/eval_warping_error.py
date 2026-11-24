#!/usr/bin/env python3
"""Evaluate object-region temporal consistency on JoliGEN inference PNGs.

Inputs, in one directory:
    000000_generated.png, 000000_orig_bbox.json, ...

For each chosen video, compute ONE fixed native-resolution crop with 25% margin that
contains every selected original object BBox. Forward/backward pretrained RAFT
flow is estimated on these identically positioned crops (no resizing).

Primary metric: squared RGB L2 error (SUM over channels, RGB in [0,1]),
averaged over pixels inside BOTH the source bbox and the forward-flow-warped
target bbox, retaining only in-bounds/forward-backward-consistent flows.

This is a CUSTOM RAFT-based object-BBox intersection metric, not the exact
implementation in either EvalCrafter or DEVIL. It does not distinguish object
pixels from background pixels inside rectangular annotations.

Examples:
    python eval_warping_error.py --self-test
    python eval_warping_error.py \\
      --input-dir /path/to/joligen/output --output-dir /path/to/eval --dry-run
    python eval_warping_error.py \\
      --input-dir /path/to/joligen/output --output-dir /path/to/eval \\
      --start-frame 0 --num-frames 16 --model large --device cuda
    python eval_warping_error.py \\
      --input-dir /path/to/joligen/output --output-dir /path/to/eval128 \\
      --num-frames 128 --model large --device cuda

Requires: torch, torchvision, numpy, pillow. No JoliGEN diffusion model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CROP_MARGIN = 0.25
FRAME_PATTERN = re.compile(r"^(\d+)_generated\.png$")


@dataclass(frozen=True)
class Frame:
    index: int
    image_path: Path
    bbox_path: Path
    bbox: tuple[int, int, int, int]  # full-frame, xyxy, half-open


def _finite_bbox(
    value: object, path: Path, width: int, height: int
) -> tuple[int, int, int, int]:
    """Read exact JoliGEN *_orig_bbox.json coords; reject ambiguous/misaligned boxes."""
    if isinstance(value, dict):
        if all(k in value for k in ("x0", "y0", "x1", "y1")):
            value = [value[k] for k in ("x0", "y0", "x1", "y1")]
        elif all(k in value for k in ("xmin", "ymin", "xmax", "ymax")):
            value = [value[k] for k in ("xmin", "ymin", "xmax", "ymax")]
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{path}: expected [x0,y0,x1,y1] or a recognized dict")
    if any(
        not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
        for v in value
    ):
        raise ValueError(f"{path}: bbox coordinates must be finite numbers: {value}")
    # Outward rounding ensures integer pixel crop contains full fractional bbox.
    x0, y0 = math.floor(value[0]), math.floor(value[1])
    x1, y1 = math.ceil(value[2]), math.ceil(value[3])
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            f"{path}: invalid/out-of-image xyxy bbox {value}; frame={width}x{height}"
        )
    return x0, y0, x1, y1


def discover_frames(
    input_dir: Path, start: int, count: int
) -> tuple[list[Frame], tuple[int, int]]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if start < 0 or count < 2:
        raise ValueError("--start-frame must be >=0 and --num-frames must be >=2")
    images: dict[int, Path] = {}
    for path in input_dir.iterdir():
        match = FRAME_PATTERN.fullmatch(path.name)
        if match:
            index = int(match.group(1))
            if index in images:
                raise ValueError(
                    f"Duplicate frame index {index}: {images[index]} and {path}"
                )
            images[index] = path
    indices = list(range(start, start + count))
    missing = [i for i in indices if i not in images]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} generated PNG(s), e.g. {missing[:8]}. "
            f"Required exact numeric indices {start}..{start + count - 1}; found {len(images)} total."
        )
    dimensions = None
    selected: list[Frame] = []
    for i in indices:
        img_path = images[i]
        bbox_path = input_dir / f"{i:06d}_orig_bbox.json"
        if not bbox_path.is_file():
            raise FileNotFoundError(f"Missing bbox JSON for frame {i}: {bbox_path}")
        with Image.open(img_path) as im:
            if im.mode not in ("RGB", "RGBA"):
                raise ValueError(
                    f"Expected RGB/RGBA generated PNG, got {im.mode}: {img_path}"
                )
            wh = im.size
        if dimensions is None:
            dimensions = wh
        elif wh != dimensions:
            raise ValueError(
                f"Inconsistent generated image dimensions: {img_path}: {wh}; expected {dimensions}"
            )
        bbox = _finite_bbox(json.loads(bbox_path.read_text()), bbox_path, *wh)
        selected.append(Frame(i, img_path, bbox_path, bbox))
    assert dimensions is not None
    return selected, dimensions


def fixed_crop(
    boxes: list[tuple[int, int, int, int]], wh: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Cover all selected boxes with 25% union-size margin on each side."""
    if not boxes:
        raise ValueError("No boxes")
    w, h = wh
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN
    return (
        max(0, math.floor(x0 - mx)),
        max(0, math.floor(y0 - my)),
        min(w, math.ceil(x1 + mx)),
        min(h, math.ceil(y1 + my)),
    )


def raft_padding(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    return max(128, math.ceil(w / 8) * 8) - w, max(128, math.ceil(h / 8) * 8) - h


def read_crop_as_uint8(frame: Frame, rect: tuple[int, int, int, int]) -> torch.Tensor:
    with Image.open(frame.image_path) as im:
        arr = np.array(im.convert("RGB").crop(rect), copy=True)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW RGB uint8 CPU


def box_mask(
    box: tuple[int, int, int, int],
    crop: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """[H,W] bool, exact half-open integer xyxy coordinates on native crop."""
    x0, y0, x1, y1 = box
    left, top, right, bottom = crop
    h, w = bottom - top, right - left
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    mask[y0 - top : y1 - top, x0 - left : x1 - left] = True
    return mask


def _flow_grid(
    flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """flow [2,H,W] in source-pixel units -> sample grid and finite in-bounds mask."""
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"Expected flow [2,H,W], got {tuple(flow.shape)}")
    h, w = flow.shape[-2:]
    if w <= 1 or h <= 1:
        raise ValueError("grid_sample align_corners=True requires H,W>1")
    finite = torch.isfinite(flow).all(dim=0)
    flow_safe = torch.where(torch.isfinite(flow), flow, torch.zeros_like(flow))
    ys, xs = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=torch.float32),
        torch.arange(w, device=flow.device, dtype=torch.float32),
        indexing="ij",
    )
    sample_x = xs + flow_safe[0].float()
    sample_y = ys + flow_safe[1].float()
    inbounds = (
        finite
        & (sample_x >= 0)
        & (sample_x <= w - 1)
        & (sample_y >= 0)
        & (sample_y <= h - 1)
    )
    grid = torch.stack(
        (2 * sample_x / (w - 1) - 1, 2 * sample_y / (h - 1) - 1), dim=-1
    )[None]
    return grid, inbounds, sample_x, sample_y


def sample_with_flow(
    tensor: torch.Tensor, grid: torch.Tensor, *, mode: str
) -> torch.Tensor:
    """Sample target [C,H,W] at p + forward_flow(p), output on source grid."""
    return F.grid_sample(
        tensor[None].float(), grid, mode=mode, padding_mode="zeros", align_corners=True
    )[0]


def evaluate_pair(
    source: torch.Tensor,
    target: torch.Tensor,
    source_bbox: torch.Tensor,
    target_bbox: torch.Tensor,
    forward: torch.Tensor,
    backward: torch.Tensor,
    *,
    fb_alpha: float = 0.01,
    fb_beta: float = 0.5,
    return_diagnostics: bool = False,
) -> tuple[dict, dict | None]:
    """CUDA/CPU tensor-only metric; requires no RAFT model and is synthetic-testable.

    source/target [3,H,W] float [0,1], boxes [H,W] bool, flows [2,H,W]
    (full native-pixel units). All tensors on same device.
    """
    if source.shape != target.shape or source.ndim != 3 or source.shape[0] != 3:
        raise ValueError("source and target must be RGB [3,H,W] with identical size")
    if forward.shape != backward.shape or forward.shape != (2, *source.shape[1:]):
        raise ValueError("Flows must be [2,H,W] matching images")
    if source_bbox.shape != source.shape[1:] or target_bbox.shape != source_bbox.shape:
        raise ValueError("BBox mask shapes must equal [H,W]")
    if fb_alpha < 0 or fb_beta < 0:
        raise ValueError("FB thresholds must be >=0")

    grid, inbounds, _, _ = _flow_grid(forward)
    warp_target = sample_with_flow(target, grid, mode="bilinear")
    warped_target_box = (
        sample_with_flow(target_bbox[None], grid, mode="nearest")[0] > 0.5
    )

    backward_finite = torch.isfinite(backward).all(dim=0)
    backward_safe = torch.where(
        torch.isfinite(backward), backward, torch.zeros_like(backward)
    )
    backward_at_target = sample_with_flow(backward_safe, grid, mode="bilinear")
    # Require the full bilinear footprint of the backward flow to be finite.
    backward_finite_at_target = sample_with_flow(
        backward_finite[None], grid, mode="bilinear"
    )[0] >= (1.0 - 1e-6)
    fb_sq = (forward + backward_at_target).square().sum(dim=0)
    flow_sq = forward.square().sum(dim=0) + backward_at_target.square().sum(dim=0)
    fb_ok = (
        backward_finite_at_target
        & torch.isfinite(fb_sq)
        & (fb_sq <= fb_alpha * flow_sq + fb_beta)
    )
    valid = inbounds & fb_ok

    diff_sq_rgb = (source - warp_target).square().sum(dim=0)  # sum of 3 RGB channels
    diff_mae_rgb = (source - warp_target).abs().mean(dim=0)
    intersection_before_validity = source_bbox & warped_target_box & inbounds
    intersection = intersection_before_validity & valid
    source_only = source_bbox & valid
    union = (source_bbox | warped_target_box) & valid

    def score(mask: torch.Tensor) -> tuple[float, int, float, float]:
        count = int(mask.sum().item())
        if not count:
            return math.nan, 0, math.nan, math.nan
        # Explicitly divide by NUM PIXELS, not RGB channels (primary sum convention).
        mse_rgb_sum = float(diff_sq_rgb[mask].mean().item())
        mse_rgb_mean = mse_rgb_sum / 3.0
        mae_rgb = float(diff_mae_rgb[mask].mean().item())
        return mse_rgb_sum, count, mse_rgb_mean, mae_rgb

    primary, n_primary, primary_mean, primary_mae = score(intersection)
    source_we, n_source, _, _ = score(source_only)
    full_we, n_full, _, _ = score(valid)
    union_we, n_union, _, _ = score(union)
    n_box = int(source_bbox.sum().item())
    n_candidate = int(intersection_before_validity.sum().item())
    n_oob_box = int((source_bbox & ~inbounds).sum().item())
    n_fb_rejected = int((intersection_before_validity & ~fb_ok).sum().item())
    flow_roi = torch.linalg.vector_norm(forward.float(), dim=0)
    flow_mask = source_bbox & valid
    mean_roi_flow = (
        float(flow_roi[flow_mask].mean().item()) if flow_mask.any().item() else math.nan
    )
    row = {
        "roi_warp_mse_rgb_sum": primary,
        "roi_warp_mse_rgb_mean": primary_mean,
        "roi_warp_mae_rgb_mean": primary_mae,
        "bbox_t_warp_mse_rgb_sum": source_we,
        "union_warp_mse_rgb_sum": union_we,
        "full_crop_warp_mse_rgb_sum": full_we,
        "mean_roi_flow_px": mean_roi_flow,
        "bbox_t_pixels": n_box,
        "candidate_correspondence_pixels": n_candidate,
        "valid_intersection_pixels": n_primary,
        "valid_bbox_t_pixels": n_source,
        "valid_union_pixels": n_union,
        "valid_full_crop_pixels": n_full,
        "out_of_bounds_bbox_t_pixels": n_oob_box,
        "fb_rejected_candidate_pixels": n_fb_rejected,
        "roi_region_coverage": n_candidate / n_box if n_box else math.nan,
        "flow_valid_coverage": n_primary / n_candidate if n_candidate else math.nan,
        "roi_total_valid_coverage": n_primary / n_box if n_box else math.nan,
        "status": "valid" if n_primary else "invalid_no_correspondences",
    }
    diagnostics = None
    if return_diagnostics:
        diagnostics = {
            "source": source.detach(),
            "target": target.detach(),
            "warped_target": warp_target.detach(),
            "intersection_mask": intersection.detach(),
            "flow_valid_mask": valid.detach(),
            "error_rgb_sum": diff_sq_rgb.detach(),
        }
    return row, diagnostics


def init_raft(size: str, device: torch.device):
    """Torchvision pretrained RAFT; no diffusion/ONNX dependencies."""
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    if size == "large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights).to(device).eval()
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights).to(device).eval()
    return model, weights.transforms(), str(weights)


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available: run with --device cpu or use a CUDA server"
        )
    return torch.device(name)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No CSV rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _nanmean(rows: list[dict], key: str) -> float:
    values = [float(r[key]) for r in rows if math.isfinite(float(r[key]))]
    return sum(values) / len(values) if values else math.nan


def _save_diagnostics(out: Path, t: int, diag: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def img(tensor: torch.Tensor) -> Image.Image:
        return Image.fromarray(
            (tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255)
            .round()
            .astype(np.uint8)
        )

    img(diag["source"]).save(out / f"{t:06d}_source.png")
    img(diag["warped_target"]).save(out / f"{t:06d}_warped_target.png")
    for key in ("intersection_mask", "flow_valid_mask"):
        Image.fromarray(diag[key].detach().cpu().to(torch.uint8).numpy() * 255).save(
            out / f"{t:06d}_{key}.png"
        )
    # RMSE of RGB channels per pixel, mapped [0,1] -> [0,255] grayscale.
    error = (diag["error_rgb_sum"] / 3).sqrt().detach().cpu().clamp(0, 1).numpy()
    Image.fromarray((error * 255).round().astype(np.uint8)).save(
        out / f"{t:06d}_rmse_map.png"
    )


def evaluate_video(
    frames: list[Frame],
    wh: tuple[int, int],
    rect: tuple[int, int, int, int],
    out: Path,
    *,
    device: torch.device,
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    preprocessing: Callable,
    fb_alpha: float,
    fb_beta: float,
    diagnostics_count: int,
) -> list[dict]:
    """model callback accepts preprocessed [1,3,H,W] pairs and returns [1,2,H,W]."""
    rows: list[dict] = []
    first_raw = read_crop_as_uint8(frames[0], rect)
    first_box = box_mask(frames[0].bbox, rect, device)
    for i, (a, b) in enumerate(zip(frames, frames[1:])):
        second_raw = read_crop_as_uint8(b, rect)
        second_box = box_mask(b.bbox, rect, device)
        # Torchvision's weights.transforms expects uint8 [0,255] or float [0,1],
        # and maps to the RAFT-required [-1,1]. Images used for scoring stay [0,1].
        img1_raw = first_raw[None]
        img2_raw = second_raw[None]
        pad_w, pad_h = raft_padding(rect)
        img1_padded = F.pad(img1_raw, (0, pad_w, 0, pad_h), mode="replicate")
        img2_padded = F.pad(img2_raw, (0, pad_w, 0, pad_h), mode="replicate")
        raft1, raft2 = preprocessing(img1_padded, img2_padded)
        raft1, raft2 = raft1.to(device), raft2.to(device)
        with torch.inference_mode():
            forward = model(raft1, raft2)[
                0, :, : first_raw.shape[1], : first_raw.shape[2]
            ].float()
            backward = model(raft2, raft1)[
                0, :, : first_raw.shape[1], : first_raw.shape[2]
            ].float()
            src = first_raw.to(device, dtype=torch.float32) / 255.0
            dst = second_raw.to(device, dtype=torch.float32) / 255.0
            values, diag = evaluate_pair(
                src,
                dst,
                first_box,
                second_box,
                forward,
                backward,
                fb_alpha=fb_alpha,
                fb_beta=fb_beta,
                return_diagnostics=i < diagnostics_count,
            )
        row = {"frame_t": a.index, "frame_t1": b.index, **values}
        rows.append(row)
        if diag is not None:
            _save_diagnostics(out / "diagnostics", a.index, diag)
        print(
            f"{a.index:06d}->{b.index:06d} "
            f"WE={values['roi_warp_mse_rgb_sum']:.7g} "
            f"valid={values['valid_intersection_pixels']}/"
            f"{values['candidate_correspondence_pixels']} "
            f"status={values['status']}",
            flush=True,
        )
        first_raw, first_box = second_raw, second_box
        del raft1, raft2, forward, backward, src, dst, diag
    return rows


def summarize(
    rows: list[dict],
    frames: list[Frame],
    rect: tuple[int, int, int, int],
    input_dir: Path,
    weight: str,
) -> dict:
    valid = [r for r in rows if r["status"] == "valid"]
    return {
        "input_dir": str(input_dir.resolve()),
        "start_frame": frames[0].index,
        "end_frame": frames[-1].index,
        "num_frames": len(frames),
        "transitions": len(rows),
        "valid_transitions": len(valid),
        "invalid_transitions": len(rows) - len(valid),
        "fixed_crop_xyxy": json.dumps(rect),
        "raft_weights": weight,
        "mean_roi_warp_mse_rgb_sum": _nanmean(valid, "roi_warp_mse_rgb_sum"),
        "mean_roi_warp_mse_rgb_mean": _nanmean(valid, "roi_warp_mse_rgb_mean"),
        "mean_bbox_t_warp_mse_rgb_sum": _nanmean(rows, "bbox_t_warp_mse_rgb_sum"),
        "mean_union_warp_mse_rgb_sum": _nanmean(rows, "union_warp_mse_rgb_sum"),
        "mean_full_crop_warp_mse_rgb_sum": _nanmean(rows, "full_crop_warp_mse_rgb_sum"),
        "mean_roi_flow_px": _nanmean(rows, "mean_roi_flow_px"),
        "mean_roi_region_coverage": _nanmean(rows, "roi_region_coverage"),
        "mean_flow_valid_coverage": _nanmean(rows, "flow_valid_coverage"),
        "mean_roi_total_valid_coverage": _nanmean(rows, "roi_total_valid_coverage"),
    }


def self_test() -> None:
    """Deterministic CPU tests: no pretrained weights, no CUDA/model download."""
    cpu = torch.device("cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    # Fixed-crop selection with moving boxes and different sizes.
    boxes = [(80, 90, 115, 130), (160, 100, 205, 155), (140, 210, 210, 260)]
    rect = fixed_crop(boxes, (700, 600))
    assert rect == (47, 47, 243, 303), rect
    assert fixed_crop([(0, 0, 20, 20), (500, 0, 530, 20)], (700, 600)) == (
        0,
        0,
        663,
        25,
    )
    assert raft_padding((0, 0, 129, 73)) == (7, 55)
    # Exact known horizontal translation +2 px, no occlusion.
    h = w = 16
    src = torch.zeros((3, h, w))
    src[:, 3:10, 3:8] = torch.tensor([0.4, 0.6, 0.8])[:, None, None]
    dst = torch.zeros_like(src)
    dst[:, :, 2:] = src[:, :, :-2]
    sbox = torch.zeros(h, w, dtype=torch.bool)
    tbox = torch.zeros_like(sbox)
    sbox[3:10, 3:8] = True
    tbox[3:10, 5:10] = True
    fw = torch.zeros((2, h, w))
    fw[0] = 2.0
    bw = torch.zeros_like(fw)
    bw[0] = -2.0
    r, _ = evaluate_pair(src, dst, sbox, tbox, fw, bw)
    assert r["valid_intersection_pixels"] == 35, r
    assert abs(r["roi_warp_mse_rgb_sum"]) < 1e-10, r
    assert r["mean_roi_flow_px"] == 2.0
    assert r["roi_region_coverage"] == 1.0
    # Color drift: +0.2 red in the corresponding object region.
    drift = dst.clone()
    drift[0, 3:10, 5:10] += 0.2
    r, _ = evaluate_pair(src, drift, sbox, tbox, fw, bw)
    assert abs(r["roi_warp_mse_rgb_sum"] - 0.04) < 1e-6, r
    assert abs(r["roi_warp_mse_rgb_mean"] - 0.04 / 3) < 1e-6
    # Inconsistent backward flow -> candidate all invalid (not scored as zero).
    invalid_bw = torch.zeros_like(bw)
    r, _ = evaluate_pair(src, dst, sbox, tbox, fw, invalid_bw)
    assert r["status"].startswith("invalid") and math.isnan(r["roi_warp_mse_rgb_sum"])
    # No common mask correspondence -> invalid.
    r, _ = evaluate_pair(src, dst, sbox, torch.zeros_like(tbox), fw, bw)
    assert r["candidate_correspondence_pixels"] == 0 and math.isnan(
        r["roi_warp_mse_rgb_sum"]
    )
    # Out-of-image correspondence -> no valid pixels, despite matching bbox mask.
    huge_fw = torch.zeros_like(fw)
    huge_fw[0] = 100
    r, _ = evaluate_pair(src, dst, sbox, sbox, huge_fw, bw)
    assert (
        r["valid_intersection_pixels"] == 0 and r["out_of_bounds_bbox_t_pixels"] == 35
    )
    # CSV & numeric pairing from 3 PNGs; run a mock flow model on native size.
    with tempfile.TemporaryDirectory() as t:
        folder = Path(t)
        for i in range(3):
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), "RGB").save(
                folder / f"{i:06d}_generated.png"
            )
            (folder / f"{i:06d}_orig_bbox.json").write_text(json.dumps([5, 6, 15, 16]))
        frames, wh = discover_frames(folder, 0, 3)
        rect = fixed_crop([f.bbox for f in frames], wh)

        def mock_model(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.zeros((1, 2, a.shape[-2], a.shape[-1]), device=a.device)

        rows = evaluate_video(
            frames,
            wh,
            rect,
            folder,
            device=cpu,
            model=mock_model,
            preprocessing=lambda x, y: (x.float(), y.float()),
            fb_alpha=0.01,
            fb_beta=0.5,
            diagnostics_count=0,
        )
        assert [(r["frame_t"], r["frame_t1"]) for r in rows] == [(0, 1), (1, 2)]
        assert all(
            r["status"] == "valid" and r["roi_warp_mse_rgb_sum"] == 0 for r in rows
        )
        csv_file = folder / "per_transition.csv"
        _write_csv(csv_file, rows)
        with csv_file.open(newline="") as f:
            assert len(list(csv.DictReader(f))) == 2
    print(
        "SELF-TEST PASSED: fixed crop, translation, RGB change, occlusion, bounds, empty masks, frame pairing, mock model, CSV"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir", type=Path, help="JoliGEN inference output folder"
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for CSV results")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=128,
        help="128 generates 127 consecutive transitions",
    )
    parser.add_argument("--model", choices=("small", "large"), default="large")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--fb-alpha",
        type=float,
        default=0.01,
        help="Forward-backward squared threshold scale",
    )
    parser.add_argument(
        "--fb-beta",
        type=float,
        default=0.5,
        help="Forward-backward squared threshold offset (px^2)",
    )
    parser.add_argument(
        "--save-diagnostics",
        type=int,
        default=0,
        metavar="N",
        help="Save debug PNGs for first N transitions (default 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all files and crop geometry; do not load RAFT",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic CPU tests without RAFT weights",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.input_dir is None or args.output_dir is None:
        parser.error("--input-dir and --output-dir are required unless --self-test")
    if (
        args.start_frame < 0
        or args.num_frames < 2
        or args.save_diagnostics < 0
        or args.fb_alpha < 0
        or args.fb_beta < 0
    ):
        parser.error(
            "Start >=0, num-frames >=2, diagnostics >=0, FB thresholds >=0 required"
        )
    frames, wh = discover_frames(args.input_dir, args.start_frame, args.num_frames)
    boxes = [f.bbox for f in frames]
    crop = fixed_crop(boxes, wh)
    print(
        f"Validated {len(frames)} generated PNGs + bbox JSONs, original dimensions={wh}"
    )
    print(
        f"Fixed native crop (25% margin, full-image xyxy): {crop}; transitions={len(frames)-1}"
    )
    settings = {
        "input_dir": str(args.input_dir.resolve()),
        "frames": [frames[0].index, frames[-1].index],
        "original_width_height": list(wh),
        "fixed_crop_xyxy": list(crop),
        "crop_width_height": [crop[2] - crop[0], crop[3] - crop[1]],
        "crop_mode": "margin",
        "crop_margin_per_side": CROP_MARGIN,
        "raft_padding_right_bottom": list(raft_padding(crop)),
        "model": args.model,
        "device_requested": args.device,
        "fb_alpha": args.fb_alpha,
        "fb_beta": args.fb_beta,
        "primary_mask": "bbox_t AND warp_nearest(bbox_t+1, forward_flow) AND inbounds AND forward_backward_consistent",
        "bbox_source": "*_orig_bbox.json (full-resolution original object annotation)",
        "photometric_error": "mean over valid pixels of sum_c (RGB_t[c]-warped_RGB_t+1[c])^2, RGB in [0,1]",
        "weights": None,
        "torch_version": torch.__version__,
        "start_index": args.start_frame,
        "num_frames": args.num_frames,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        settings["dry_run"] = True
        (args.output_dir / "config.json").write_text(
            json.dumps(settings, indent=2, default=str) + "\n"
        )
        print("DRY RUN OK (no model weights loaded, no GPU evaluation)")
        return 0
    device = get_device(args.device)
    model, preprocessing, weight_name = init_raft(args.model, device)
    settings.update({"device_actual": str(device), "weights": weight_name})
    try:
        import torchvision

        settings["torchvision_version"] = torchvision.__version__
    except ImportError:
        pass
    rows = evaluate_video(
        frames,
        wh,
        crop,
        args.output_dir,
        device=device,
        model=lambda a, b: model(a, b)[-1],
        preprocessing=preprocessing,
        fb_alpha=args.fb_alpha,
        fb_beta=args.fb_beta,
        diagnostics_count=args.save_diagnostics,
    )
    _write_csv(args.output_dir / "per_transition.csv", rows)
    summary = summarize(rows, frames, crop, args.input_dir, weight_name)
    _write_csv(args.output_dir / "summary.csv", [summary])
    (args.output_dir / "config.json").write_text(
        json.dumps(settings, indent=2, default=str) + "\n"
    )
    print(
        f"Finished: {len(rows)} transitions; {summary['invalid_transitions']} invalid"
    )
    print(
        f"Mean primary ROI WE (sum squared RGB): {summary['mean_roi_warp_mse_rgb_sum']:.8g}"
    )
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
