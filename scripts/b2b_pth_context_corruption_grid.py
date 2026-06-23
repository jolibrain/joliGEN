import argparse
import copy
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JG_DIR = os.path.join(SCRIPT_DIR, "../")
sys.path.append(SCRIPT_DIR)
sys.path.append(JG_DIR)

import b2b_onnx_denoiser_infer_autoregressive_progress_bbox as onnx_runner
import b2b_pth_denoiser_infer_autoregressive_progress_bbox as pth_runner
from b2b_export_onnx import parse_device


def default_corruption_variants():
    variants = [{"name": "clean", "kind": "clean"}]
    for name, rgb in [
        ("black", (0, 0, 0)),
        ("white", (255, 255, 255)),
        ("gray", (128, 128, 128)),
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
    ]:
        variants.append({"name": f"color_{name}", "kind": "color", "rgb": rgb})
    for std in [0.1, 0.25, 0.5]:
        variants.append({"name": f"noise_{std:g}", "kind": "noise", "std": std})
    for kernel in [5, 15, 31]:
        variants.append({"name": f"blur_k{kernel}", "kind": "blur", "kernel": kernel})
    for delta in [-0.4, -0.2, 0.2, 0.4]:
        variants.append(
            {"name": f"brightness_{delta:+g}", "kind": "brightness", "delta": delta}
        )
    for scale in [0.5, 0.75, 1.25, 1.5]:
        variants.append(
            {"name": f"contrast_{scale:g}", "kind": "contrast", "scale": scale}
        )
    variants.extend(
        [
            {"name": "shuffle_context", "kind": "shuffle"},
            {"name": "down_up_0.25", "kind": "down_up", "scale": 0.25},
            {"name": "down_up_0.5", "kind": "down_up", "scale": 0.5},
            {"name": "edges", "kind": "edges"},
        ]
    )
    return variants


def default_mask_variants(variant_set):
    variants = [{"name": "mask_clean", "kind": "mask_clean"}]
    if variant_set in ["comprehensive", "morphology"]:
        for kernel in [5, 11, 21]:
            variants.append(
                {
                    "name": f"mask_dilate_k{kernel}",
                    "kind": "morph",
                    "op": "dilate",
                    "kernel": kernel,
                }
            )
        variants.extend(
            [
                {
                    "name": "mask_close_k11",
                    "kind": "morph",
                    "op": "close",
                    "kernel": 11,
                },
                {
                    "name": "mask_close_k21",
                    "kind": "morph",
                    "op": "close",
                    "kernel": 21,
                },
            ]
        )
    if variant_set in ["comprehensive", "geometry"]:
        for scale in [1.25, 1.5, 2.0]:
            variants.append(
                {
                    "name": f"mask_scale_{scale:g}",
                    "kind": "scale",
                    "sx": scale,
                    "sy": scale,
                }
            )
        for name, sx, sy in [
            ("wide_1.5", 1.5, 1.0),
            ("tall_1.5", 1.0, 1.5),
            ("wide_2", 2.0, 1.0),
            ("tall_2", 1.0, 2.0),
        ]:
            variants.append(
                {"name": f"mask_{name}", "kind": "scale", "sx": sx, "sy": sy}
            )
        for shift in [8, 16]:
            variants.extend(
                [
                    {
                        "name": f"mask_shift_left_{shift}",
                        "kind": "shift",
                        "dx": -shift,
                        "dy": 0,
                    },
                    {
                        "name": f"mask_shift_right_{shift}",
                        "kind": "shift",
                        "dx": shift,
                        "dy": 0,
                    },
                    {
                        "name": f"mask_shift_up_{shift}",
                        "kind": "shift",
                        "dx": 0,
                        "dy": -shift,
                    },
                    {
                        "name": f"mask_shift_down_{shift}",
                        "kind": "shift",
                        "dx": 0,
                        "dy": shift,
                    },
                ]
            )
    if variant_set == "comprehensive":
        variants.extend(
            [
                {
                    "name": "mask_jitter_amp4",
                    "kind": "jitter",
                    "amplitude": 4.0,
                    "smooth": 31,
                },
                {
                    "name": "mask_jitter_amp8",
                    "kind": "jitter",
                    "amplitude": 8.0,
                    "smooth": 31,
                },
                {"name": "mask_smooth_k9", "kind": "smooth", "kernel": 9},
                {"name": "mask_smooth_k21", "kind": "smooth", "kernel": 21},
            ]
        )
    return variants


def clone_frame_data(frame_data):
    cloned = {}
    for key, value in frame_data.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def mask_tensor_to_uint8(mask):
    arr = mask.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    return (arr > 0.5).astype(np.uint8)


def uint8_to_mask_tensor(mask, reference):
    tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return tensor.to(dtype=reference.dtype, device=reference.device)


def ensure_binary_nonempty(mask, fallback):
    mask = (mask > 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return fallback.copy()
    return mask


def kernel_for(size):
    size = int(size)
    if size <= 0:
        raise ValueError(f"Kernel size must be positive, got {size}")
    if size % 2 == 0:
        size += 1
    return np.ones((size, size), dtype=np.uint8)


def contour_region(mask, width, side):
    mask_np = mask_tensor_to_uint8(mask)
    kernel = kernel_for(2 * max(1, int(width)) + 1)
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    eroded = cv2.erode(mask_np, kernel, iterations=1)
    if side == "outside":
        region = np.logical_and(dilated > 0, mask_np == 0)
    elif side == "inside":
        region = np.logical_and(mask_np > 0, eroded == 0)
    elif side == "both":
        region = np.logical_and(dilated > 0, eroded == 0)
    else:
        raise ValueError(f"Unsupported contour side: {side}")
    return uint8_to_mask_tensor(region.astype(np.uint8), mask)


def resize_mask_about_center(mask_np, sx, sy):
    ys, xs = np.nonzero(mask_np)
    if len(xs) == 0:
        return mask_np.copy()

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = mask_np[y0:y1, x0:x1]
    h, w = crop.shape
    new_w = max(1, int(round(w * float(sx))))
    new_h = max(1, int(round(h * float(sy))))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized = (resized > 0).astype(np.uint8)

    cx = float(xs.mean())
    cy = float(ys.mean())
    paste_x0 = int(round(cx - new_w / 2.0))
    paste_y0 = int(round(cy - new_h / 2.0))
    out = np.zeros_like(mask_np)
    dst_x0 = max(0, paste_x0)
    dst_y0 = max(0, paste_y0)
    dst_x1 = min(mask_np.shape[1], paste_x0 + new_w)
    dst_y1 = min(mask_np.shape[0], paste_y0 + new_h)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return mask_np.copy()
    src_x0 = dst_x0 - paste_x0
    src_y0 = dst_y0 - paste_y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = resized[src_y0:src_y1, src_x0:src_x1]
    return ensure_binary_nonempty(out, mask_np)


def shift_mask(mask_np, dx, dy):
    h, w = mask_np.shape
    matrix = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
    shifted = cv2.warpAffine(
        mask_np,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return ensure_binary_nonempty(shifted, mask_np)


def jitter_mask(mask_np, amplitude, smooth, rng):
    inside = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask_np, cv2.DIST_L2, 5)
    signed_distance = inside - outside
    noise = rng.normal(0.0, 1.0, size=mask_np.shape).astype(np.float32)
    kernel = int(smooth)
    if kernel % 2 == 0:
        kernel += 1
    noise = cv2.GaussianBlur(noise, (kernel, kernel), 0)
    max_abs = float(np.max(np.abs(noise)))
    if max_abs > 0:
        noise = noise / max_abs
    jittered = (signed_distance + float(amplitude) * noise > 0).astype(np.uint8)
    return ensure_binary_nonempty(jittered, mask_np)


def perturb_mask_tensor(mask, variant, rng):
    kind = variant["kind"]
    mask_np = mask_tensor_to_uint8(mask)
    if kind == "mask_clean":
        new_mask = mask_np
    elif kind == "morph":
        kernel = kernel_for(variant["kernel"])
        op = variant["op"]
        if op == "dilate":
            new_mask = cv2.dilate(mask_np, kernel, iterations=1)
        elif op == "erode":
            new_mask = cv2.erode(mask_np, kernel, iterations=1)
        elif op == "open":
            new_mask = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        elif op == "close":
            new_mask = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        else:
            raise ValueError(f"Unsupported morphology op: {op}")
        new_mask = ensure_binary_nonempty(new_mask, mask_np)
    elif kind == "scale":
        new_mask = resize_mask_about_center(mask_np, variant["sx"], variant["sy"])
    elif kind == "shift":
        new_mask = shift_mask(mask_np, variant["dx"], variant["dy"])
    elif kind == "jitter":
        new_mask = jitter_mask(mask_np, variant["amplitude"], variant["smooth"], rng)
    elif kind == "smooth":
        kernel = int(variant["kernel"])
        if kernel % 2 == 0:
            kernel += 1
        blurred = cv2.GaussianBlur(mask_np.astype(np.float32), (kernel, kernel), 0)
        new_mask = ensure_binary_nonempty((blurred >= 0.5).astype(np.uint8), mask_np)
    else:
        raise ValueError(f"Unknown mask perturbation kind: {kind}")
    new_mask = ensure_binary_nonempty(np.maximum(new_mask, mask_np), mask_np)
    return uint8_to_mask_tensor(new_mask, mask)


def tensor_to_hwc_float(tensor):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    return np.transpose(arr, (1, 2, 0)).astype(np.float32)


def hwc_float_to_tensor(arr, reference):
    chw = np.transpose(arr.astype(np.float32), (2, 0, 1))
    return (
        torch.from_numpy(chw)
        .unsqueeze(0)
        .to(dtype=reference.dtype, device=reference.device)
    )


def apply_image_transform(tensor, variant, rng):
    kind = variant["kind"]
    if kind == "clean":
        return tensor.clone()
    if kind == "color":
        rgb = np.asarray(variant["rgb"], dtype=np.float32) / 127.5 - 1.0
        out = torch.empty_like(tensor)
        for channel, value in enumerate(rgb):
            out[:, channel, :, :] = float(value)
        return out
    if kind == "noise":
        noise = rng.normal(
            loc=0.0, scale=float(variant["std"]), size=tuple(tensor.shape)
        ).astype(np.float32)
        return (tensor + torch.from_numpy(noise).to(tensor)).clamp(-1.0, 1.0)
    if kind == "brightness":
        return (tensor + float(variant["delta"])).clamp(-1.0, 1.0)
    if kind == "contrast":
        return (tensor * float(variant["scale"])).clamp(-1.0, 1.0)
    if kind == "blur":
        arr = tensor_to_hwc_float(tensor)
        kernel = int(variant["kernel"])
        arr = cv2.GaussianBlur(arr, (kernel, kernel), 0)
        return hwc_float_to_tensor(arr, tensor).clamp(-1.0, 1.0)
    if kind == "down_up":
        arr = tensor_to_hwc_float(tensor)
        h, w = arr.shape[:2]
        scale = float(variant["scale"])
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        small = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_AREA)
        arr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return hwc_float_to_tensor(arr, tensor).clamp(-1.0, 1.0)
    if kind == "edges":
        arr = tensor_to_hwc_float(tensor)
        uint8 = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 64, 128)
        edge_rgb = np.repeat(edges[:, :, None], 3, axis=2).astype(np.float32)
        edge_rgb = edge_rgb / 127.5 - 1.0
        return hwc_float_to_tensor(edge_rgb, tensor).clamp(-1.0, 1.0)
    raise ValueError(f"Unknown corruption kind: {kind}")


def apply_shuffle_context(tensor, context_mask, rng):
    out = tensor.clone()
    context = context_mask[0, 0].bool()
    if not bool(context.any()):
        return out
    for channel in range(out.shape[1]):
        values = out[0, channel][context].clone()
        order = torch.from_numpy(rng.permutation(values.numel())).to(values.device)
        out[0, channel][context] = values[order]
    return out


def corrupt_tensor_region(tensor, region_mask, variant, rng):
    region_mask = region_mask.to(dtype=tensor.dtype, device=tensor.device)
    if variant["kind"] == "shuffle":
        corrupted = apply_shuffle_context(tensor, region_mask, rng)
    else:
        corrupted = apply_image_transform(tensor, variant, rng)
    return tensor * (1.0 - region_mask) + corrupted * region_mask


def corrupt_frame_data_contour(frame_data, variant, rng, contour_width, contour_side):
    frame = clone_frame_data(frame_data)
    if variant["kind"] == "clean":
        return frame
    region = contour_region(frame["mask"], contour_width, contour_side)
    for key in ["y_t", "y0_tensor", "img_tensor"]:
        if frame.get(key) is not None:
            frame[key] = corrupt_tensor_region(frame[key], region, variant, rng)
    return frame


def apply_mask_variant_to_frame(frame_data, variant, rng):
    frame = clone_frame_data(frame_data)
    new_mask = perturb_mask_tensor(frame["mask"], variant, rng)
    frame["mask"] = new_mask
    if frame.get("cond_image") is not None:
        frame["cond_image"] = new_mask.clone()
    return frame


def load_two_frames(args, train_json):
    dataset_root = onnx_runner.resolve_dataset_root(
        args.paths_in_file, args.dataset_root
    )
    pairs = onnx_runner.read_paths_file(args.paths_in_file)
    if args.start_index + 2 > len(pairs):
        raise ValueError(
            f"paths.txt has {len(pairs)} entries, need at least 2 from "
            f"start_index={args.start_index}"
        )

    frames = []
    for offset, (img_rel, bbox_rel) in enumerate(
        pairs[args.start_index : args.start_index + 2]
    ):
        frame_data = onnx_runner.preprocess_with_repo_crop(
            img_path=os.path.join(dataset_root, img_rel),
            bbox_path=os.path.join(dataset_root, bbox_rel),
            bbox_index=args.bbox_index,
            train_json=train_json,
            device=torch.device("cpu"),
            crop_size_override=args.crop_size,
        )
        frame_data["index"] = args.start_index + offset
        frame_data["img_rel"] = img_rel
        frame_data["bbox_rel"] = bbox_rel
        frames.append(frame_data)
    return frames, dataset_root


def infer_second_generated_crop(
    session,
    frames,
    label,
    seed,
    train_json,
    denoise_steps,
    object_refs=None,
    temporal_frame_step=None,
):
    rng = np.random.default_rng(seed)
    _, _, _, output_h, output_w = onnx_runner.get_train_shape(train_json)
    params = onnx_runner.get_b2b_params(
        train_json,
        onnx_runner.require_square_crop_size(output_h, output_w),
    )
    prev_frame, frame_data = frames

    y0_np = prev_frame["y0_tensor"].numpy()
    sigma = rng.integers(0, 2, size=(y0_np.shape[0],)).astype(np.float32)
    sigma = sigma / 1.0 * 0.71
    sigma = sigma.reshape(y0_np.shape[0], 1, 1, 1)
    eps_ctx = rng.standard_normal(size=y0_np.shape, dtype=np.float32)
    y0_noisy_first = torch.from_numpy(y0_np + sigma * eps_ctx)

    y_t_batch = onnx_runner.prepare_tensors([prev_frame["y_t"], frame_data["y_t"]])
    mask_batch = onnx_runner.prepare_tensors([prev_frame["mask"], frame_data["mask"]])
    global_context_batch = onnx_runner.prepare_tensors(
        [prev_frame.get("global_context"), frame_data.get("global_context")]
    )
    if params["mask_as_channel"]:
        cond_image_batch = onnx_runner.prepare_tensors(
            [prev_frame["cond_image"], frame_data["cond_image"]]
        )
    else:
        cond_image_batch = onnx_runner.prepare_tensors(
            [y0_noisy_first, frame_data["cond_image"]]
        )

    labels = np.asarray(
        [onnx_runner.resolve_b2b_label(train_json, frame_data["label_cls"], label)],
        dtype=np.int64,
    )
    init_noise = rng.standard_normal(size=y_t_batch.shape, dtype=np.float32)
    out_tensor = onnx_runner.restoration_with_denoiser(
        session=session,
        y=y_t_batch.numpy().astype(np.float32),
        y_cond=(
            None
            if cond_image_batch is None
            else cond_image_batch.numpy().astype(np.float32)
        ),
        denoise_steps=denoise_steps,
        mask=mask_batch.numpy().astype(np.float32),
        labels=labels,
        params=params,
        init_noise=init_noise,
        temporal_frame_step=temporal_frame_step,
        global_context=(
            None
            if global_context_batch is None
            else global_context_batch.numpy().astype(np.float32)
        ),
        object_refs=(
            None if object_refs is None else object_refs.numpy().astype(np.float32)
        ),
    )

    out_tensor_torch = torch.from_numpy(out_tensor).squeeze(0)
    return out_tensor_torch[-1:].clone()


def draw_label(tile, label, label_height):
    out = np.full((tile.shape[0] + label_height, tile.shape[1], 3), 255, dtype=np.uint8)
    out[label_height:] = tile
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    max_width = tile.shape[1] - 8
    text = label
    while cv2.getTextSize(text, font, font_scale, thickness)[0][0] > max_width:
        if len(text) <= 4:
            break
        text = text[:-4] + "..."
    cv2.putText(
        out,
        text,
        (4, max(14, label_height - 8)),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return out


def make_grid(tiles, labels, grid_cols, label_height):
    if len(tiles) != len(labels):
        raise ValueError("tiles and labels must have the same length")
    if not tiles:
        raise ValueError("No tiles to render")

    labeled = [
        draw_label(tile, label, label_height) for tile, label in zip(tiles, labels)
    ]
    tile_h, tile_w = labeled[0].shape[:2]
    cols = max(1, int(grid_cols))
    rows = int(math.ceil(len(labeled) / cols))
    grid = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    for index, tile in enumerate(labeled):
        row = index // cols
        col = index % cols
        y0 = row * tile_h
        x0 = col * tile_w
        grid[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return grid


def mask_to_bgr_tile(mask):
    mask_np = mask_tensor_to_uint8(mask) * 255
    return cv2.cvtColor(mask_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def mask_bbox(mask_np):
    ys, xs = np.nonzero(mask_np)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def draw_mask_boundary_bbox(tile, mask, color, thickness=2):
    mask_np = mask_tensor_to_uint8(mask)
    contours, _ = cv2.findContours(
        (mask_np * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(tile, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    bbox = mask_bbox(mask_np)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(
            tile,
            (x0, y0),
            (max(x0, x1 - 1), max(y0, y1 - 1)),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return tile


def overlay_mask_change(tile, original_mask, perturbed_mask):
    out = tile.copy()
    draw_mask_boundary_bbox(out, original_mask, (0, 255, 0), thickness=1)
    draw_mask_boundary_bbox(out, perturbed_mask, (0, 0, 255), thickness=2)
    return out


def output_name_for_group(args, group_name):
    if args.study_mode != "both" and args.output_name != "corruption_grid.png":
        return args.output_name
    if group_name == "contour":
        return (
            "contour_corruption_grid.png"
            if args.study_mode == "both"
            else args.output_name
        )
    if group_name == "mask":
        return "mask_perturbation_grid.png"
    raise ValueError(f"Unknown output group: {group_name}")


def run_variant_grid(
    session,
    frames,
    variants,
    transform_frames,
    args,
    train_json,
    denoise_steps,
    grid_name,
    manifest_key,
    save_mask_tiles=False,
    overlay_mask_boundaries=False,
    object_refs=None,
):
    tiles = []
    labels = []
    mask_tiles = []
    entries = []
    for variant_index, variant in enumerate(variants):
        variant_seed = int(args.seed) + 1009 * variant_index
        rng = np.random.default_rng(variant_seed)
        transformed_frames = transform_frames(variant, rng)
        generated = infer_second_generated_crop(
            session=session,
            frames=transformed_frames,
            label=args.label,
            seed=args.seed,
            train_json=train_json,
            denoise_steps=denoise_steps,
            object_refs=object_refs,
            temporal_frame_step=args.temporal_frame_step,
        )
        tile = onnx_runner.chw_to_bgr_uint8(generated)
        if overlay_mask_boundaries:
            tile = overlay_mask_change(
                tile,
                original_mask=frames[-1]["mask"],
                perturbed_mask=transformed_frames[-1]["mask"],
            )
        tiles.append(tile)
        labels.append(variant["name"])
        if save_mask_tiles:
            mask_tiles.append(mask_to_bgr_tile(transformed_frames[-1]["mask"]))
        entries.append(
            {
                "index": variant_index,
                "name": variant["name"],
                "params": variant,
                "variant_seed": variant_seed,
                "grid_row": variant_index // max(1, args.grid_cols),
                "grid_col": variant_index % max(1, args.grid_cols),
            }
        )
        print(f"[{manifest_key} {variant_index + 1}/{len(variants)}] {variant['name']}")

    grid_path = os.path.join(args.output_dir, grid_name)
    cv2.imwrite(grid_path, make_grid(tiles, labels, args.grid_cols, args.label_height))
    mask_grid_path = None
    if save_mask_tiles:
        mask_grid_path = os.path.join(
            args.output_dir, "mask_perturbation_masks_grid.png"
        )
        cv2.imwrite(
            mask_grid_path,
            make_grid(mask_tiles, labels, args.grid_cols, args.label_height),
        )
    return {
        "grid": grid_path,
        "mask_grid": mask_grid_path,
        "variants": entries,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_in_file",
        required=True,
        help="Path to a JoliGEN B2B checkpoint, e.g. latest_net_G_A_ema.pth",
    )
    parser.add_argument(
        "--paths_in_file",
        required=True,
        help="paths.txt containing 'image_rel_path bbox_rel_path' pairs",
    )
    parser.add_argument(
        "--dataset_root",
        help="Dataset root used to resolve relative paths. Defaults to dirname(dirname(paths_in_file))",
    )
    parser.add_argument(
        "--train_config",
        help="Optional train_config.json. Defaults to the checkpoint directory.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument(
        "--start_index", type=int, default=0, help="Start line index in paths.txt"
    )
    parser.add_argument(
        "--bbox_index", type=int, default=0, help="Which bbox line to use"
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        help=(
            "Override the square crop window size before resizing to the model input "
            "size. Useful for debugging crop context."
        ),
    )
    parser.add_argument("--label", type=int, default=None, help="Override class label")
    parser.add_argument("--seed", type=int, default=0, help="Seed for all runs")
    parser.add_argument(
        "--temporal_frame_step",
        type=float,
        help=(
            "Raw temporal frame stride for checkpoints trained with "
            "alg.b2b_temporal_frame_step_conditioning. Defaults to "
            "data.temporal_frame_step from train_config.json, then 1."
        ),
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        help="Override denoise step count. Defaults to first entry from alg.b2b_denoise_timesteps",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use the sibling *_ema.pth checkpoint instead of model_in_file",
    )
    parser.add_argument(
        "--alg_b2b_object_ref_paths",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Override static object reference image paths for checkpoints trained "
            "with alg.b2b_object_ref_paths."
        ),
    )
    parser.add_argument(
        "--study_mode",
        choices=["contour", "mask", "both"],
        default="contour",
        help="Run contour-band pixel corruption, mask perturbations, or both",
    )
    parser.add_argument(
        "--contour_width",
        type=int,
        default=12,
        help="Width in pixels for the mask contour corruption band",
    )
    parser.add_argument(
        "--contour_side",
        choices=["outside", "inside", "both"],
        default="outside",
        help="Which side of the mask boundary to use for contour corruption",
    )
    parser.add_argument(
        "--mask_variant_set",
        choices=["comprehensive", "geometry", "morphology"],
        default="comprehensive",
        help="Mask perturbation family to run",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="Also write a grid of the perturbed masks for mask study modes",
    )
    parser.add_argument(
        "--grid_cols", type=int, default=5, help="Number of columns in the output grid"
    )
    parser.add_argument(
        "--label_height",
        type=int,
        default=24,
        help="Pixel height reserved for each tile label",
    )
    parser.add_argument(
        "--output_name",
        default="corruption_grid.png",
        help="Grid image filename inside output_dir",
    )
    parser.add_argument(
        "--manifest_name",
        default="manifest.json",
        help="Manifest filename inside output_dir",
    )
    parser.add_argument(
        "--max_variants",
        type=int,
        help="Optional smoke-test limit on the number of variants to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = parse_device(args.device)
    session, weights_path, train_config_path = pth_runner.load_pth_session(
        args.model_in_file, args.train_config, device, args.use_ema
    )
    train_json, _ = onnx_runner.load_train_config(args.model_in_file, train_config_path)
    if train_json.get("alg", {}).get("diffusion_cond_image_creation", "y_t") != "y_t":
        raise NotImplementedError(
            "This runner currently supports only alg.diffusion_cond_image_creation = 'y_t'."
        )

    frames, dataset_root = load_two_frames(args, train_json)
    denoise_steps = onnx_runner.resolve_denoise_steps(train_json, args.denoise_steps)
    object_refs = onnx_runner.load_object_refs_for_inference(
        train_json, args.alg_b2b_object_ref_paths
    )
    if args.max_variants is not None and args.max_variants <= 0:
        raise ValueError("--max_variants must be positive when provided")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "model_in_file": args.model_in_file,
        "checkpoint": weights_path,
        "train_config": train_config_path,
        "paths_in_file": args.paths_in_file,
        "dataset_root": dataset_root,
        "start_index": args.start_index,
        "bbox_index": args.bbox_index,
        "crop_size_override": args.crop_size,
        "seed": args.seed,
        "denoise_steps": denoise_steps,
        "temporal_frame_step": onnx_runner.resolve_temporal_frame_step(
            train_json, args.temporal_frame_step
        ),
        "global_context_mode": onnx_runner.b2b_global_context_mode_from_train_json(
            train_json
        ),
        "global_context_conditioning": (
            onnx_runner.b2b_global_context_enabled_from_train_json(train_json)
        ),
        "object_ref_count": 0 if object_refs is None else int(object_refs.shape[0]),
        "study_mode": args.study_mode,
        "contour_width": args.contour_width,
        "contour_side": args.contour_side,
        "mask_variant_set": args.mask_variant_set,
        "mask_overlay": {
            "enabled_for_mask_study": True,
            "original_mask_bgr": [0, 255, 0],
            "perturbed_mask_bgr": [0, 0, 255],
        },
        "grid_cols": args.grid_cols,
        "outputs": {},
    }

    if args.study_mode in ["contour", "both"]:
        contour_variants = default_corruption_variants()
        if args.max_variants is not None:
            contour_variants = contour_variants[: args.max_variants]

        def transform_contour(variant, rng):
            return [
                corrupt_frame_data_contour(
                    frame,
                    variant,
                    rng,
                    contour_width=args.contour_width,
                    contour_side=args.contour_side,
                )
                for frame in frames
            ]

        contour_result = run_variant_grid(
            session=session,
            frames=frames,
            variants=contour_variants,
            transform_frames=transform_contour,
            args=args,
            train_json=train_json,
            denoise_steps=denoise_steps,
            grid_name=output_name_for_group(args, "contour"),
            manifest_key="contour",
            object_refs=object_refs,
        )
        manifest["outputs"]["contour"] = contour_result

    if args.study_mode in ["mask", "both"]:
        mask_variants = default_mask_variants(args.mask_variant_set)
        if args.max_variants is not None:
            mask_variants = mask_variants[: args.max_variants]

        def transform_mask(variant, rng):
            return [
                apply_mask_variant_to_frame(frame, variant, rng) for frame in frames
            ]

        mask_result = run_variant_grid(
            session=session,
            frames=frames,
            variants=mask_variants,
            transform_frames=transform_mask,
            args=args,
            train_json=train_json,
            denoise_steps=denoise_steps,
            grid_name=output_name_for_group(args, "mask"),
            manifest_key="mask",
            save_mask_tiles=args.save_masks,
            overlay_mask_boundaries=True,
            object_refs=object_refs,
        )
        manifest["outputs"]["mask"] = mask_result

    manifest_path = os.path.join(args.output_dir, args.manifest_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"checkpoint   : {weights_path}")
    print(f"train_config : {train_config_path}")
    print(f"device       : {device}")
    print(
        "temporal_frame_step: "
        f"{onnx_runner.resolve_temporal_frame_step(train_json, args.temporal_frame_step)}"
    )
    print(
        "object_refs  : " f"{0 if object_refs is None else int(object_refs.shape[0])}"
    )
    for group_name, group_result in manifest["outputs"].items():
        print(f"{group_name}_variants: {len(group_result['variants'])}")
        print(f"{group_name}_grid    : {group_result['grid']}")
        if group_result.get("mask_grid"):
            print(f"{group_name}_masks   : {group_result['mask_grid']}")
    print(f"manifest     : {manifest_path}")


if __name__ == "__main__":
    main()
