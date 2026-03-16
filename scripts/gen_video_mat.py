import argparse
import math
import os
import random
import re
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

jg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(jg_dir)

from data.online_creation import crop_image
from scripts.gen_single_image_mat import (
    load_mat_model,
    prepare_mat_batch,
    resolve_mask_delta,
    tensor_to_bgr_uint8,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="predict", help="inference process name")
    parser.add_argument(
        "--model_in_file",
        required=True,
        help="file path to MAT generator weights (.pth file)",
    )
    parser.add_argument(
        "--dataroot",
        required=True,
        help="text file containing image/mask pairs, one per line",
    )
    parser.add_argument(
        "--dir_out",
        required=True,
        help="directory where the generated video and optional frames are written",
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        help="optional prefix prepended to the image and mask paths from --dataroot",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        help="image width, defaults to the MAT crop size from train_config.json",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        help="image height, defaults to the MAT crop size from train_config.json",
    )
    parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
    parser.add_argument("--gpuid", type=int, default=0, help="which GPU to use")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="concatenate original, masked input, and output in the generated video",
    )
    parser.add_argument("--fps", default=30, type=int, help="video fps")
    parser.add_argument(
        "--nb_img_max", default=10000, type=int, help="max number of frames to render"
    )
    parser.add_argument(
        "--sv_frames", action="store_true", help="whether to save generated frames"
    )
    parser.add_argument(
        "--start_frame",
        default=-1,
        type=int,
        help="if >= 0, start from this frame index in the input list",
    )
    parser.add_argument(
        "--bbox_ref_id",
        type=int,
        default=-1,
        help="bbox id to use when a frame mask path points to a bbox .txt file",
    )
    return parser.parse_args()


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def resolve_path(data_prefix, path):
    if os.path.isabs(path) or data_prefix == "":
        return path
    return os.path.join(data_prefix, path)


def load_frame_specs(dataroot):
    frame_specs = []
    with open(dataroot, "r") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                continue
            elts = line.split()
            if len(elts) != 2:
                raise ValueError(
                    f"Each line in {dataroot} must contain exactly two paths, got: {line!r}"
                )
            frame_specs.append((elts[0], elts[1]))
    return frame_specs


def load_bbox_entries(mask_in, select_cat):
    bbox_entries = []
    with open(mask_in, "r") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"{mask_in} contains an invalid bbox line: {line!r}")
            cat = int(parts[0])
            if select_cat != -1 and cat != select_cat:
                continue
            bbox_entries.append([cat, int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])])
    if len(bbox_entries) == 0:
        raise ValueError(f"No bbox found in {mask_in}")
    return bbox_entries


def resolve_bbox_ref_id(opt, mask_in, bbox_ref_id):
    bbox_entries = load_bbox_entries(
        mask_in, getattr(opt, "data_online_select_category", -1)
    )
    if bbox_ref_id != -1:
        if bbox_ref_id < 0 or bbox_ref_id >= len(bbox_entries):
            raise ValueError(
                f"bbox_ref_id={bbox_ref_id} is out of range for {mask_in} ({len(bbox_entries)} bbox entries)"
            )
        return bbox_ref_id, bbox_entries
    return random.randrange(len(bbox_entries)), bbox_entries


def _get_loaded_image_size(img_path, load_size):
    with Image.open(img_path) as image:
        orig_width, orig_height = image.size

    if load_size != []:
        resolved = list(load_size)
        if len(resolved) == 1:
            resolved = [resolved[0], resolved[0]]
        loaded_width, loaded_height = resolved[0], resolved[1]
    else:
        loaded_width, loaded_height = orig_width, orig_height

    ratio_x = loaded_width / orig_width
    ratio_y = loaded_height / orig_height

    return (orig_width, orig_height), (loaded_width, loaded_height), ratio_x, ratio_y


def _apply_mask_delta(mask_delta, cat, bbox_width, bbox_height):
    if mask_delta == [[]]:
        return 0, 0

    if len(mask_delta) == 1:
        mask_delta_cat = mask_delta[0]
    else:
        if len(mask_delta) <= cat - 1:
            raise ValueError("too few classes, can't find mask_delta value")
        mask_delta_cat = mask_delta[cat - 1]

    if isinstance(mask_delta[0][0], float):
        if len(mask_delta_cat) == 1:
            delta_x = mask_delta_cat[0] * bbox_width
            delta_y = mask_delta_cat[0] * bbox_height
        else:
            delta_x = mask_delta_cat[0] * bbox_width
            delta_y = mask_delta_cat[1] * bbox_height
    elif isinstance(mask_delta[0][0], int):
        if len(mask_delta_cat) == 1:
            delta_x = mask_delta_cat[0]
            delta_y = mask_delta_cat[0]
        else:
            delta_x = mask_delta_cat[0]
            delta_y = mask_delta_cat[1]
    else:
        raise ValueError("mask_delta value is incorrect.")

    return int(delta_x), int(delta_y)


def compute_bbox_insert_region(opt, img_in, mask_in, bbox_ref_id, bbox_entries):
    (orig_width, orig_height), (loaded_width, loaded_height), ratio_x, ratio_y = (
        _get_loaded_image_size(
            img_in, getattr(opt, "data_online_creation_load_size_A", [])
        )
    )

    mask_random_offset = [0.0]
    crop_coordinates = crop_image(
        img_path=img_in,
        bbox_path=mask_in,
        mask_random_offset=mask_random_offset,
        mask_delta=resolve_mask_delta(opt),
        crop_delta=0,
        mask_square=getattr(opt, "data_online_creation_mask_square_A", False),
        crop_dim=getattr(opt, "data_online_creation_crop_size_A", opt.data_crop_size),
        output_dim=opt.data_crop_size,
        context_pixels=getattr(opt, "data_online_context_pixels", 0),
        load_size=getattr(opt, "data_online_creation_load_size_A", []),
        get_crop_coordinates=True,
        crop_center=True,
        select_cat=getattr(opt, "data_online_select_category", -1),
        fixed_mask_size=getattr(opt, "data_online_fixed_mask_size", -1),
        bbox_ref_id=bbox_ref_id,
        inverted_mask=getattr(opt, "data_inverted_mask", False),
        single_bbox=getattr(opt, "data_online_single_bbox", False),
        random_bbox=False,
    )

    cat, xmin, ymin, xmax, ymax = bbox_entries[bbox_ref_id]
    xmin = math.floor(xmin * ratio_x)
    ymin = math.floor(ymin * ratio_y)
    xmax = math.floor(xmax * ratio_x)
    ymax = math.floor(ymax * ratio_y)

    delta_x, delta_y = _apply_mask_delta(
        resolve_mask_delta(opt), cat, xmax - xmin, ymax - ymin
    )
    xmin -= delta_x
    xmax += delta_x
    ymin -= delta_y
    ymax += delta_y

    if getattr(opt, "data_online_creation_mask_square_A", False):
        sdiff = (xmax - xmin) - (ymax - ymin)
        if sdiff > 0:
            ymax += int(sdiff / 2)
            ymin -= int(sdiff / 2)
        else:
            xmax += -int(sdiff / 2)
            xmin -= -int(sdiff / 2)

    fixed_mask_size = getattr(opt, "data_online_fixed_mask_size", -1)
    if fixed_mask_size > 0:
        xdiff = fixed_mask_size - (xmax - xmin)
        ydiff = fixed_mask_size - (ymax - ymin)
        ymax += int(ydiff / 2)
        ymin -= int(ydiff / 2)
        xmax += int(xdiff / 2)
        xmin -= int(xdiff / 2)

    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(xmax, loaded_width)
    ymax = min(ymax, loaded_height)

    x_crop_rel, y_crop_rel, crop_size = crop_coordinates
    context_pixels = getattr(opt, "data_online_context_pixels", 0)
    x_crop = x_crop_rel + xmin
    y_crop = y_crop_rel + ymin

    if x_crop < context_pixels:
        x_crop = context_pixels
    if x_crop + crop_size + context_pixels > loaded_width:
        x_crop = loaded_width - (crop_size + context_pixels)
    if y_crop < context_pixels:
        y_crop = context_pixels
    if y_crop + crop_size + context_pixels > loaded_height:
        y_crop = loaded_height - (crop_size + context_pixels)

    crop_box = [
        x_crop - context_pixels,
        y_crop - context_pixels,
        x_crop + crop_size + context_pixels,
        y_crop + crop_size + context_pixels,
    ]

    crop_box = [
        max(0, min(orig_width, round(crop_box[0] / ratio_x))),
        max(0, min(orig_height, round(crop_box[1] / ratio_y))),
        max(0, min(orig_width, round(crop_box[2] / ratio_x))),
        max(0, min(orig_height, round(crop_box[3] / ratio_y))),
    ]

    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError(f"Invalid crop box computed for {img_in}: {crop_box}")

    return crop_box


def load_original_frame(img_in):
    frame = cv2.imread(img_in)
    if frame is None:
        raise ValueError(f"Could not read input frame {img_in}")
    return frame


def generate_frame(model, opt, device, img_in, mask_in, img_width, img_height, args):
    bbox_ref_id = args.bbox_ref_id
    bbox_insert_region = None
    original_frame = None
    if mask_in.endswith(".txt"):
        bbox_ref_id, bbox_entries = resolve_bbox_ref_id(opt, mask_in, bbox_ref_id)
        bbox_insert_region = compute_bbox_insert_region(
            opt, img_in, mask_in, bbox_ref_id, bbox_entries
        )
        original_frame = load_original_frame(img_in)

    batch, original_np, _ = prepare_mat_batch(
        opt,
        img_in,
        mask_in,
        img_width,
        img_height,
        device,
        bbox_ref_id=bbox_ref_id,
    )
    model.set_input(batch)
    model.inference(1)
    output_bgr = tensor_to_bgr_uint8(model.fake_B)

    if bbox_insert_region is not None:
        x0, y0, x1, y1 = bbox_insert_region
        full_output = original_frame.copy()
        resized_crop = cv2.resize(
            output_bgr,
            (x1 - x0, y1 - y0),
            interpolation=cv2.INTER_CUBIC,
        )
        full_output[y0:y1, x0:x1] = resized_crop
        if not args.compare:
            return full_output
        return np.concatenate([original_frame, full_output], axis=1)

    if not args.compare:
        return output_bgr

    input_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
    return np.concatenate([input_bgr, output_bgr], axis=1)


def save_frame(frame, dir_out, name, idx, num_frames):
    out_name = f"{name}_{str(idx).zfill(len(str(max(1, num_frames))))}.jpg"
    cv2.imwrite(os.path.join(dir_out, out_name), frame)


def run_video_inference(args):
    model, opt, device = load_mat_model(args.model_in_file, args.cpu, args.gpuid)

    img_width = args.img_width if args.img_width is not None else opt.data_crop_size
    img_height = args.img_height if args.img_height is not None else opt.data_crop_size

    frame_specs = load_frame_specs(args.dataroot)
    frame_specs.sort(key=lambda paths: natural_keys(paths[0]))
    if args.start_frame >= 0:
        frame_specs = frame_specs[args.start_frame :]
    frame_specs = frame_specs[: args.nb_img_max]

    if len(frame_specs) == 0:
        raise ValueError(f"No valid image/mask pairs found in {args.dataroot}")

    os.makedirs(args.dir_out, exist_ok=True)
    video_path = os.path.join(args.dir_out, args.name + "_generated_video.avi")
    out = None
    video_size = None
    total_generation_time = 0.0
    num_generated = 0
    try:
        with torch.inference_mode():
            progress = tqdm(
                enumerate(frame_specs), total=len(frame_specs), desc="Generating video"
            )
            for idx, (image, mask) in progress:
                img_in = resolve_path(args.data_prefix, image)
                mask_in = resolve_path(args.data_prefix, mask)
                start_time = time.perf_counter()
                frame = generate_frame(
                    model, opt, device, img_in, mask_in, img_width, img_height, args
                )
                if out is None:
                    video_size = (frame.shape[1], frame.shape[0])
                    out = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                        args.fps,
                        video_size,
                    )
                    if not out.isOpened():
                        raise RuntimeError(f"Could not open video writer for {video_path}")
                elif (frame.shape[1], frame.shape[0]) != video_size:
                    raise ValueError(
                        f"Frame size {(frame.shape[1], frame.shape[0])} does not match video size {video_size}"
                    )
                generation_time = time.perf_counter() - start_time
                total_generation_time += generation_time
                num_generated += 1
                progress.set_postfix(
                    avg_sec_per_image=total_generation_time / num_generated
                )
                if args.sv_frames:
                    save_frame(frame, args.dir_out, args.name, idx, len(frame_specs))
                out.write(frame)
    finally:
        if out is not None:
            out.release()

    args.total_generation_time = total_generation_time
    args.num_generated_images = num_generated
    args.avg_generation_time_per_image = (
        total_generation_time / num_generated if num_generated > 0 else 0.0
    )
    print(
        f"Average generation time per image: {args.avg_generation_time_per_image:.4f}s"
    )

    return video_path


if __name__ == "__main__":
    run_video_inference(parse_args())
