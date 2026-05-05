import argparse
import copy
import json
import math
import os
import random
import re
import sys
import tempfile
import warnings
import logging

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

import re
from collections import defaultdict
from PIL import Image

jg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(jg_dir)
from segment_anything import SamPredictor

from data.online_creation import crop_image, fill_mask_with_color, fill_mask_with_random
from models import diffusion_networks
from models.modules.diffusion_utils import set_new_noise_schedule
from models.modules.sam.sam_inference import (
    compute_mask_with_sam,
    init_sam_net,
    load_sam_weight,
    predict_sam_mask,
)
from models.modules.utils import download_sam_weight
from options.inference_diffusion_options import InferenceDiffusionOptions
from options.train_options import TrainOptions
from util.mask_generation import (
    fill_img_with_canny,
    fill_img_with_depth,
    fill_img_with_hed,
    fill_img_with_hough,
    fill_img_with_sam,
    fill_img_with_sketch,
)
from util.script import get_override_options_names
from util.util import flatten_json
import time


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def prepare_tensors(tensor_list):
    if all(tensor is not None for tensor in tensor_list):
        return torch.stack(tensor_list, dim=0).permute(1, 0, 2, 3, 4)
    return None


def prepare_label_tensor(label_list, device):
    labels = torch.tensor(
        [int(label) for label in label_list], dtype=torch.long, device=device
    )
    return labels.unsqueeze(0)


def separate_tensors(tensor):
    if tensor is not None and tensor.shape[0] == 1:
        # Remove the batch dimension
        tensor = tensor.squeeze(0)  # Shape becomes [8, 3, 64, 64]
        # Split the tensor into individual images
        image_list = list(tensor.unbind(0))  # Extract 8 images as individual tensors
        image_list = [img.unsqueeze(0) for img in image_list]
        return image_list
    return None


def load_model(
    model_in_dir,
    model_in_filename,
    device,
    sampling_steps,
    sampling_method,
    model_prior_321_backwardcompatibility,
):
    train_json_path = model_in_dir + "/train_config.json"
    with open(train_json_path, "r") as jsonf:
        train_json = json.load(jsonf)
    train_json["gpu_ids"] = str(device.index)
    opt = TrainOptions().parse_json(train_json)
    opt.jg_dir = jg_dir
    if opt.data_online_creation_mask_random_offset_A != [0.0]:
        warnings.warn(
            f"disabling data_online_creation_mask_random_offset_A in inference mode"
        )
        opt.data_online_creation_mask_random_offset_A = [0.0]

    opt.model_prior_321_backwardcompatibility = model_prior_321_backwardcompatibility
    if opt.model_type in ["cm", "cm_gan", "b2b"]:
        opt.alg_palette_sampling_method = sampling_method
        opt.alg_diffusion_cond_embed_dim = 256
    model = diffusion_networks.define_G(opt=opt, **vars(opt))
    model.eval()

    # handle old models
    weights = torch.load(
        os.path.join(model_in_dir, model_in_filename), map_location=torch.device(device)
    )
    if opt.model_prior_321_backwardcompatibility:
        weights = {
            k.replace("denoise_fn.cond_embed", "cond_embed"): v
            for k, v in weights.items()
        }
    if not any(k.startswith("denoise_fn.model") for k in weights.keys()):
        weights = {
            k.replace("denoise_fn", "denoise_fn.model"): v for k, v in weights.items()
        }
    if not any(k.startswith("denoise_fn.netl_embedder_") for k in weights.keys()):
        weights = {
            k.replace("l_embedder_", "denoise_fn.netl_embedder_"): v
            for k, v in weights.items()
        }
    model.load_state_dict(weights, strict=False)

    # sampling steps
    if sampling_steps > 0:
        model.denoise_fn.model.beta_schedule["test"]["n_timestep"] = sampling_steps
        set_new_noise_schedule(model.denoise_fn.model, "test")

    if opt.model_type == "palette":
        model.set_new_sampling_method(sampling_method)

    if opt.alg_diffusion_task == "pix2pix":
        opt.alg_diffusion_cond_image_creation = "pix2pix"

    model = model.to(device)
    return model, opt


def to_np(img):
    img = img.detach().data.cpu().float().numpy()[0]
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def cond_augment(cond, rotation, persp_horizontal, persp_vertical):
    cond = Image.fromarray(cond)
    cond = transforms.RandomRotation(rotation, expand=True)(cond)
    w, h = cond.size
    startpoints = [[0, 0], [w, 0], [w, h], [0, h]]
    endpoints = copy.deepcopy(startpoints)
    # horizontal perspective
    d = h * persp_horizontal * random.random()
    if random.choice([True, False]):
        # seen from left
        endpoints[1][1] += d
        endpoints[2][1] -= d
    else:
        # seen from right
        endpoints[0][1] += d
        endpoints[3][1] -= d
    # vertical perspective
    d = h * persp_vertical * random.random()
    if random.choice([True, False]):
        # seen from above
        endpoints[3][0] += d
        endpoints[2][0] -= d
    else:
        # seen from below
        endpoints[0][0] += d
        endpoints[1][0] -= d
    cond = cond.crop(cond.getbbox())
    cond = transforms.functional.perspective(cond, startpoints, endpoints)
    return np.array(cond)


def generate_streaming(
    seed,
    model_in_file,
    lmodel,
    lopt,
    cpu,
    gpuid,
    sampling_steps,
    img_in,
    mask_in,
    ref_in,
    bbox_in,
    cond_in,
    cond_keep_ratio,
    bbox_width_factor,
    bbox_height_factor,
    bbox_ref_id,
    crop_width,
    crop_height,
    img_width,
    img_height,
    paths_in_file,
    dir_out,
    write,
    previous_frame,
    name,
    mask_delta,
    mask_square,
    sampling_method,
    cond_rotation,
    cond_persp_horizontal,
    cond_persp_vertical,
    alg_diffusion_cond_image_creation,
    alg_diffusion_sketch_canny_thresholds,
    cls,
    alg_diffusion_super_resolution_downsample,
    alg_diffusion_guidance_scale,
    data_refined_mask,
    min_crop_bbox_ratio,
    alg_palette_ddim_num_steps,
    alg_b2b_denoise_timesteps,
    alg_palette_ddim_eta,
    model_prior_321_backwardcompatibility,
    logger,
    iteration,
    nb_samples,
    compare=False,
    vid_fps=18,
    **unused_options,
):
    PROGRESS_NUM_STEPS = 4
    if seed >= 0:
        torch.manual_seed(seed)

    if not cpu:
        device = torch.device("cuda:" + str(gpuid))
    else:
        device = torch.device("cpu")

    if lmodel is None:
        model, opt = load_model(
            os.path.dirname(model_in_file),
            os.path.basename(model_in_file),
            device,
            sampling_steps,
            sampling_method,
            model_prior_321_backwardcompatibility,
        )
    else:
        model = lmodel
        opt = lopt

    if alg_diffusion_cond_image_creation is not None:
        opt.alg_diffusion_cond_image_creation = alg_diffusion_cond_image_creation

    if logger:
        logger.info(
            f"[it: %i/%i] - [1/%i] model loaded"
            % (iteration, nb_samples, PROGRESS_NUM_STEPS)
        )

    conditioning = opt.alg_diffusion_cond_embed

    for i, delta_values in enumerate(mask_delta):
        if len(delta_values) == 1:
            mask_delta[i].append(delta_values[0])

    with open(paths_in_file, "r") as file:
        lines = file.readlines()
    image_bbox_pairs = []
    for line in lines:
        parts = line.strip().split()
        image_bbox_pairs.append((parts[0], parts[1]))
    image_bbox_pairs.sort(key=lambda x: natural_keys(x[0]), reverse=False)

    limited_paths_img = [pair[0] for pair in image_bbox_pairs]
    limited_paths_bbox = [pair[1] for pair in image_bbox_pairs]

    N = limited_paths_img
    N_bbox = limited_paths_bbox

    seq_half = 1
    if seq_half != 1:
        raise ValueError("Streaming mode currently supports seq_half=1 only")

    select_canny_list = [
        1 if i in {0, (len(N) - 1) // 2, len(N) - 1} else 0 for i in range(len(N))
    ]

    inputmix = True
    prev_frame = None
    y0_noisy_first = None
    last_seq_half_y_t = None
    last_seq_half_cond_image = None
    last_seq_half_y0_tensor = None
    last_seq_half_mask = None
    last_seq_half_cls = None
    ref_tensor = None
    cls_tensor = None

    # noise parameters
    num_buckets = 10
    max_sigma = 0.71

    total_generation_time = 0.0
    num_generated_images = 0
    frames_written = 0

    video_path = os.path.join(dir_out, f"{name}_generated_video.avi")
    video_writer = None
    video_size = None
    b2b_mask_condition = opt.model_type == "b2b" and getattr(
        opt, "alg_b2b_mask_as_channel", False
    )

    def write_frame(frame_index, out_img, frame_data, cond_image_for_output):
        nonlocal num_generated_images, frames_written, video_writer, video_size
        img_orig = frame_data["img_orig"]
        y_t = frame_data["y_t"]
        mask = frame_data["mask"]
        bbox_select = frame_data["bbox_select"]
        img_tensor = frame_data["img_tensor"]
        bbox = frame_data.get("bbox")
        generated_bbox = frame_data.get("generated_bbox")
        has_bbox = frame_data["has_bbox"]
        ref_orig = frame_data.get("ref_orig")
        noisy_first = frame_data.get("noisy_first")
        noisy_first_sigma = frame_data.get("noisy_first_sigma")

        out_img_for_paste = out_img
        mask_np_for_paste = None

        if mask is not None:
            mask_np = mask.detach().float().cpu().numpy()
            if mask_np.ndim == 4:
                mask_np = mask_np[0, 0]
            elif mask_np.ndim == 3:
                mask_np = mask_np[0]
            mask_np_for_paste = (mask_np > 0).astype(np.uint8)

        if has_bbox:
            out_img_resized = cv2.resize(
                out_img_for_paste,
                (
                    min(img_orig.shape[1], bbox_select[2] - bbox_select[0]),
                    min(img_orig.shape[0], bbox_select[3] - bbox_select[1]),
                ),
            )
            out_img_real_size = img_orig.copy()
        else:
            out_img_resized = out_img_for_paste
            out_img_real_size = img_orig.copy()

        # fill out crop into original image (respect mask if available)
        if has_bbox:
            y0, y1 = bbox_select[1], bbox_select[3]
            x0, x1 = bbox_select[0], bbox_select[2]
            if mask_np_for_paste is not None:
                mask_resized = cv2.resize(
                    mask_np_for_paste,
                    (x1 - x0, y1 - y0),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_3 = mask_resized.astype(bool)[:, :, None]
                orig_crop = out_img_real_size[y0:y1, x0:x1].copy()
                out_img_real_size[y0:y1, x0:x1] = np.where(
                    mask_3, out_img_resized, orig_crop
                )
            else:
                out_img_real_size[y0:y1, x0:x1] = out_img_resized

        cond_img = None
        if cond_image_for_output is not None:
            cond_img = to_np(cond_image_for_output)

        name_out = f"{frame_index:06d}"

        if write:
            if opt.data_image_bits > 8:
                img_tensor_tmp = [frame_index]
                img_np = to_np(img_tensor_tmp)
                cv2.imwrite(os.path.join(dir_out, name_out + "_orig.png"), img_np)
                if cond_img is not None:
                    cv2.imwrite(os.path.join(dir_out, name_out + "_cond.png"), cond_img)
                cv2.imwrite(
                    os.path.join(dir_out, name_out + "_generated.png"), out_img_resized
                )
            else:
                cv2.imwrite(os.path.join(dir_out, name_out + "_orig.png"), img_orig)
                if cond_img is not None:
                    cv2.imwrite(os.path.join(dir_out, name_out + "_cond.png"), cond_img)
                cv2.imwrite(
                    os.path.join(dir_out, name_out + "_generated.png"),
                    out_img_real_size,
                )
                cv2.imwrite(os.path.join(dir_out, name_out + "_y_t.png"), to_np(y_t))
                if mask is not None:
                    cv2.imwrite(
                        os.path.join(dir_out, name_out + "_y_0.png"), to_np(img_tensor)
                    )
                    cv2.imwrite(
                        os.path.join(dir_out, name_out + "_generated_crop.png"),
                        out_img_for_paste,
                    )
                    cv2.imwrite(
                        os.path.join(dir_out, name_out + "_mask.png"), to_np(mask)
                    )
                if ref_orig is not None:
                    cv2.imwrite(
                        os.path.join(dir_out, name_out + "_ref_orig.png"), ref_orig
                    )
                if cond_in and has_bbox:
                    orig_crop = img_orig[
                        bbox_select[1] : bbox_select[3],
                        bbox_select[0] : bbox_select[2],
                    ]
                    cv2.imwrite(
                        os.path.join(dir_out, name_out + "_orig_crop.png"), orig_crop
                    )
                if has_bbox and bbox is not None:
                    with open(
                        os.path.join(dir_out, name_out + "_orig_bbox.json"), "w"
                    ) as out:
                        out.write(json.dumps(bbox))
                if generated_bbox:
                    with open(
                        os.path.join(dir_out, name_out + "_generated_bbox.json"), "w"
                    ) as out:
                        out.write(json.dumps(generated_bbox))
                if noisy_first is not None:
                    noisy_first_crop = np.clip(to_np(noisy_first), 0, 255).astype(
                        np.uint8
                    )
                    cv2.imwrite(
                        os.path.join(dir_out, name_out + "_noisy_first_crop.png"),
                        noisy_first_crop,
                    )

                    noisy_first_full = noisy_first_crop
                    if has_bbox:
                        noisy_first_resized = cv2.resize(
                            noisy_first_crop,
                            (
                                min(
                                    img_orig.shape[1],
                                    bbox_select[2] - bbox_select[0],
                                ),
                                min(
                                    img_orig.shape[0],
                                    bbox_select[3] - bbox_select[1],
                                ),
                            ),
                        )
                        noisy_first_full = img_orig.copy()
                        y0, y1 = bbox_select[1], bbox_select[3]
                        x0, x1 = bbox_select[0], bbox_select[2]
                        if mask_np_for_paste is not None:
                            mask_resized = cv2.resize(
                                mask_np_for_paste,
                                (x1 - x0, y1 - y0),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            mask_3 = mask_resized.astype(bool)[:, :, None]
                            orig_crop = noisy_first_full[y0:y1, x0:x1].copy()
                            noisy_first_full[y0:y1, x0:x1] = np.where(
                                mask_3, noisy_first_resized, orig_crop
                            )
                        else:
                            noisy_first_full[y0:y1, x0:x1] = noisy_first_resized

                    cv2.imwrite(
                        os.path.join(dir_out, name_out + "_noisy_first.png"),
                        noisy_first_full,
                    )
                    if noisy_first_sigma is not None:
                        with open(
                            os.path.join(dir_out, name_out + "_noisy_first_sigma.json"),
                            "w",
                        ) as out:
                            out.write(json.dumps(noisy_first_sigma))

        video_frame = (
            np.concatenate([out_img_real_size, img_orig], axis=1)
            if compare
            else out_img_real_size
        )

        if video_writer is None:
            video_size = (video_frame.shape[1], video_frame.shape[0])
            video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                vid_fps,
                video_size,
            )
            if not video_writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {video_path}")
        elif (video_frame.shape[1], video_frame.shape[0]) != video_size:
            raise ValueError(
                f"Frame size {(video_frame.shape[1], video_frame.shape[0])} does not match video size {video_size}"
            )
        video_writer.write(video_frame)

        num_generated_images += 1
        frames_written += 1

    frame_iter = tqdm(enumerate(zip(N, N_bbox)), total=len(N), desc="Generating frames")
    try:
        for sequence_count, (img_path, bbox_path) in frame_iter:
            start_time = time.perf_counter()
            frames_written = 0
            img_in = os.path.join(
                os.path.dirname(os.path.dirname(paths_in_file)), img_path
            )
            bbox_in_path = os.path.join(
                os.path.dirname(os.path.dirname(paths_in_file)), bbox_path
            )
            frame_cls = cls if cls > 0 else 1
            bbox_select = None
            bbox = None
            generated_bbox = None

            # reading image
            if opt.data_image_bits > 8:
                img = Image.open(img_in)
                img_orig = None
            else:
                img = cv2.imread(img_in)
                img_orig = img.copy()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # reading the mask
            mask = None
            if mask_in:
                mask = cv2.imread(mask_in, 0)

            # reading reference image
            ref = None
            ref_orig = None
            if ref_in:
                ref = cv2.imread(ref_in)
                ref_orig = ref.copy()
                ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

            bboxes = []
            bbox_classes = []
            if bbox_in_path:
                with open(bbox_in_path, "r") as bboxf:
                    for line in bboxf:
                        elts = line.rstrip().split()
                        bbox_classes.append(int(elts[0]))
                        bboxes.append(
                            [int(elts[1]), int(elts[2]), int(elts[3]), int(elts[4])]
                        )

                if bbox_ref_id == -1:
                    bbox_idx = random.choice(range(len(bboxes)))
                    bbox_idx = bbox_ref_id
                if cls <= 0 and bbox_classes:
                    frame_cls = bbox_classes[bbox_idx]

                if crop_width > 0 or crop_height > 0:
                    hc_width = int(crop_width / 2)
                    hc_height = int(crop_height / 2)
                    bbox_orig = bboxes[bbox_idx]
                    if bbox_width_factor > 0.0:
                        bbox_orig[0] -= max(0, int(bbox_width_factor * bbox_orig[0]))
                        bbox_orig[2] += max(0, int(bbox_width_factor * bbox_orig[2]))
                    if bbox_height_factor > 0.0:
                        bbox_orig[1] -= max(0, int(bbox_height_factor * bbox_orig[1]))
                        bbox_orig[3] += max(0, int(bbox_height_factor * bbox_orig[3]))

                    bbox_select = bbox_orig.copy()
                    bbox_select[0] -= max(0, hc_width)
                    bbox_select[0] = max(0, bbox_select[0])
                    bbox_select[1] -= max(0, hc_height)
                    bbox_select[1] = max(0, bbox_select[1])
                    bbox_select[2] += hc_width
                    bbox_select[2] = min(img.shape[1], bbox_select[2])
                    bbox_select[3] += hc_height
                    bbox_select[3] = min(img.shape[0], bbox_select[3])
                else:
                    bbox = bboxes[bbox_idx]

                crop_coordinates = crop_image(
                    img_path=img_in,
                    bbox_path=bbox_in_path,
                    mask_delta=mask_delta,
                    mask_random_offset=opt.data_online_creation_mask_random_offset_A,
                    crop_delta=0,
                    mask_square=mask_square,
                    crop_dim=opt.data_online_creation_crop_size_A,
                    output_dim=opt.data_load_size,
                    context_pixels=opt.data_online_context_pixels,
                    load_size=opt.data_online_creation_load_size_A,
                    get_crop_coordinates=True,
                    crop_center=True,
                    bbox_ref_id=bbox_idx,
                    min_crop_bbox_ratio=min_crop_bbox_ratio,
                )

                img, mask, ref_bbox, bbox_ref_id = crop_image(
                    img_path=img_in,
                    bbox_path=bbox_in_path,
                    mask_delta=mask_delta,
                    mask_random_offset=opt.data_online_creation_mask_random_offset_A,
                    crop_delta=0,
                    mask_square=mask_square,
                    crop_dim=opt.data_online_creation_crop_size_A,
                    output_dim=opt.data_load_size,
                    context_pixels=opt.data_online_context_pixels,
                    load_size=opt.data_online_creation_load_size_A,
                    crop_coordinates=crop_coordinates,
                    crop_center=True,
                    bbox_ref_id=bbox_idx,
                    override_class=frame_cls,
                )
                x_crop, y_crop, crop_size = crop_coordinates

                bbox = bboxes[bbox_idx]
                bbox_select = bbox.copy()
                if len(mask_delta) == 1:
                    index_cls = 0
                else:
                    index_cls = int(frame_cls) - 1

                if not isinstance(mask_delta[0][0], float):
                    bbox_select[0] -= mask_delta[index_cls][0]
                    bbox_select[1] -= mask_delta[index_cls][1]
                    bbox_select[2] += mask_delta[index_cls][0]
                    bbox_select[3] += mask_delta[index_cls][1]
                else:
                    bbox_select[0] *= 1 + mask_delta[index_cls][0]
                    bbox_select[1] *= 1 + mask_delta[index_cls][1]
                    bbox_select[2] *= 1 + mask_delta[index_cls][0]
                    bbox_select[3] *= 1 + mask_delta[index_cls][1]

                if mask_square:
                    sdiff = (bbox_select[2] - bbox_select[0]) - (
                        bbox_select[3] - bbox_select[1]
                    )
                    if sdiff > 0:
                        bbox_select[3] += int(sdiff / 2)
                        bbox_select[1] -= int(sdiff / 2)
                    else:
                        bbox_select[2] += -int(sdiff / 2)
                        bbox_select[0] -= -int(sdiff / 2)

                bbox_select[1] += y_crop
                bbox_select[0] += x_crop

                bbox_select[3] = bbox_select[1] + crop_size
                bbox_select[2] = bbox_select[0] + crop_size

                bbox_select[1] -= opt.data_online_context_pixels
                bbox_select[0] -= opt.data_online_context_pixels

                bbox_select[3] += opt.data_online_context_pixels
                bbox_select[2] += opt.data_online_context_pixels

                img, mask = np.array(img), np.array(mask)

            if img_width > 0 and img_height > 0:
                if opt.data_image_bits > 8:
                    print(
                        "Requested image size differs from training crop size, resizing is not supported for images with more than 8 bits per channel"
                    )
                    exit(1)
                img = cv2.resize(img, (img_width, img_height))
                if mask is not None:
                    mask = cv2.resize(
                        mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST
                    )
                if ref is not None:
                    ref = cv2.resize(ref, (img_width, img_height))

            # insert cond image into original image
            if cond_in:
                generated_bbox = bbox
                mask_bbox = Image.fromarray(mask).getbbox()
                x0, y0, x1, y1 = mask_bbox
                bbox_w = x1 - x0
                bbox_h = y1 - y0
                cond = cv2.imread(cond_in)
                cond = cv2.cvtColor(cond, cv2.COLOR_RGB2BGR)
                cond = cond_augment(
                    cond,
                    cond_rotation,
                    cond_persp_horizontal,
                    cond_persp_vertical,
                )
                if cond_keep_ratio:
                    # pad cond image to match bbox aspect ratio
                    bbox_ratio = bbox_w / bbox_h
                    new_w = cond_w = cond.shape[1]
                    new_h = cond_h = cond.shape[0]
                    cond_ratio = cond_w / cond_h
                    if cond_ratio < bbox_ratio:
                        new_w = round(cond_w * bbox_ratio / cond_ratio)
                    elif cond_ratio > bbox_ratio:
                        new_h = round(cond_h * cond_ratio / bbox_ratio)
                    cond_pad = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    x = (new_w - cond_w) // 2
                    y = (new_h - cond_h) // 2
                    cond_pad[y : y + cond_h, x : x + cond_w] = cond
                    cond = cond_pad
                    # bbox inside mask
                    generated_bbox = [
                        (x, y),
                        (x + cond_w, y + cond_h),
                    ]
                    # bbox inside crop
                    generated_bbox = [
                        (
                            x0 + x * bbox_w / cond.shape[1],
                            y0 + y * bbox_h / cond.shape[0],
                        )
                        for x, y in generated_bbox
                    ]
                    # bbox inside original image
                    real_width = min(img_orig.shape[1], bbox_select[2] - bbox_select[0])
                    real_height = min(
                        img_orig.shape[0], bbox_select[3] - bbox_select[1]
                    )
                    generated_bbox = [
                        (
                            bbox_select[0] + x * real_width / img.shape[1],
                            bbox_select[1] + y * real_height / img.shape[0],
                        )
                        for x, y in generated_bbox
                    ]
                    # round & flatten
                    generated_bbox = list(
                        map(round, generated_bbox[0] + generated_bbox[1])
                    )

                # add 1 pixel margin for sketches
                cond = cv2.resize(
                    cond, (bbox_w - 2, bbox_h - 2), interpolation=cv2.INTER_CUBIC
                )
                cond = np.pad(cond, [(1, 1), (1, 1), (0, 0)])
                img[y0:y1, x0:x1] = cond

            # preprocessing to torch
            totensor = transforms.ToTensor()
            if opt.data_image_bits > 8:
                tranlist = [totensor, torchvision.transforms.v2.ToDtype(torch.float32)]
                bit_scaling = 2**opt.data_image_bits - 1
                tranlist += [
                    transforms.Lambda(lambda img: img * (1 / float(bit_scaling)))
                ]
                tranlist += [transforms.Normalize((0.5,), (0.5,))]
            else:
                tranlist = [
                    totensor,
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]

            tran = transforms.Compose(tranlist)
            img_tensor = tran(img).clone().detach()

            if mask is not None:
                mask = torch.from_numpy(np.array(mask, dtype=np.int64)).unsqueeze(0)
            if ref is not None:
                ref = cv2.resize(
                    ref, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC
                )
                ref_tensor = tran(ref).clone().detach()

            if not cpu:
                img_tensor = img_tensor.to(device).clone().detach()
                if mask is not None:
                    mask = mask.to(device).clone().detach()
                if ref is not None:
                    ref_tensor = ref_tensor.to(device).clone().detach()

            if mask is not None:
                if data_refined_mask or opt.data_refined_mask:
                    opt.f_s_weight_sam = "../" + opt.f_s_weight_sam
                    sam_model, _ = init_sam_net(
                        model_type_sam=opt.model_type_sam,
                        model_path=opt.f_s_weight_sam,
                        device=device,
                    )
                    mask = compute_mask_with_sam(
                        img_tensor, mask, sam_model, device, batched=False
                    ).unsqueeze(0)

                if opt.data_inverted_mask:
                    mask[mask > 0] = 2
                    mask[mask == 0] = 1
                    mask[mask == 2] = 0

                if opt.data_online_creation_rand_mask_A:
                    y_t = fill_mask_with_random(
                        img_tensor.clone().detach(), mask.clone().detach(), -1
                    )
                elif opt.data_online_creation_color_mask_A:
                    y_t = fill_mask_with_color(
                        img_tensor.clone().detach(), mask.clone().detach(), {}
                    )
            else:
                y_t = torch.randn_like(img_tensor)

            if opt.alg_diffusion_cond_image_creation == "previous_frame":
                if previous_frame is not None:
                    if isinstance(previous_frame, str):
                        previous_frame = cv2.imread(previous_frame)
                    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB)
                    previous_frame = previous_frame[
                        bbox_select[1] : bbox_select[3], bbox_select[0] : bbox_select[2]
                    ]
                    previous_frame = cv2.resize(
                        previous_frame, (opt.data_load_size, opt.data_load_size)
                    )
                    previous_frame = tran(previous_frame)
                    previous_frame = (
                        previous_frame.to(device).clone().detach().unsqueeze(0)
                    )

                    cond_image = previous_frame
                else:
                    cond_image = -1 * torch.ones_like(
                        y_t.unsqueeze(0), device=y_t.device
                    )
            elif opt.alg_diffusion_cond_image_creation == "y_t":
                if opt.model_type == "palette":
                    cond_image = y_t.unsqueeze(0)
                elif opt.model_type == "b2b" and opt.alg_b2b_mask_as_channel:
                    # For B2B mask-as-channel checkpoints, cond_image is the
                    # 1-channel mask condition that will be concatenated with
                    # the RGB stream inside the generator.
                    cond_image = mask.to(dtype=y_t.dtype).unsqueeze(0)
                else:
                    cond_image = None
            elif opt.alg_diffusion_cond_image_creation == "sketch":
                cond_image = fill_img_with_sketch(
                    img_tensor.unsqueeze(0), mask.unsqueeze(0)
                )
            elif opt.alg_diffusion_cond_image_creation == "computed_sketch":
                clamp = torch.clamp(mask, 0, 1)
                if cond_in:
                    img_tensor_canny = clamp * img_tensor + clamp - 1
                else:
                    img_tensor_canny = img_tensor
                cond_image = fill_img_with_canny(
                    img_tensor_canny.unsqueeze(0),
                    mask.unsqueeze(0),
                    low_threshold=alg_diffusion_sketch_canny_thresholds[0],
                    high_threshold=alg_diffusion_sketch_canny_thresholds[1],
                    low_threshold_random=-1,
                    high_threshold_random=-1,
                    select_canny=[select_canny_list[sequence_count]],
                )
                if cond_in:
                    cond_image = cond_image * clamp + img_tensor * (1 - clamp)
            elif opt.alg_diffusion_cond_image_creation == "sam":
                opt.f_s_weight_sam = "../" + opt.f_s_weight_sam
                if not os.path.exists(opt.f_s_weight_sam):
                    download_sam_weight(opt.f_s_weight_sam)
                sam, _ = load_sam_weight(opt.f_s_weight_sam)
                sam = sam.to(device)
                cond_image = fill_img_with_sam(
                    img_tensor.unsqueeze(0), mask.unsqueeze(0), sam, opt
                )
            elif opt.alg_diffusion_cond_image_creation == "hed":
                cond_image = fill_img_with_hed(
                    img_tensor.unsqueeze(0), mask.unsqueeze(0)
                )
            elif opt.alg_diffusion_cond_image_creation == "hough":
                cond_image = fill_img_with_hough(
                    img_tensor.unsqueeze(0), mask.unsqueeze(0)
                )
            elif opt.alg_diffusion_cond_image_creation == "depth":
                cond_image = fill_img_with_depth(
                    img_tensor.unsqueeze(0), mask.unsqueeze(0)
                )
            elif opt.alg_diffusion_cond_image_creation == "low_res":
                if alg_diffusion_super_resolution_downsample:
                    data_crop_size_low_res = int(
                        opt.data_crop_size / opt.alg_diffusion_super_resolution_scale
                    )
                    transform_lr = T.Resize(
                        (data_crop_size_low_res, data_crop_size_low_res)
                    )
                    cond_image = transform_lr(img_tensor.unsqueeze(0)).detach()
                else:
                    cond_image = img_tensor.unsqueeze(0).clone().detach()
                transform_hr = T.Resize((opt.data_crop_size, opt.data_crop_size))
                cond_image = transform_hr(cond_image).detach()
            elif opt.alg_diffusion_cond_image_creation == "pix2pix":
                if (img_height > 0 and img_height != opt.data_crop_size) or (
                    img_width > 0 and img_width != opt.data_crop_size
                ):
                    transform_hr = T.Resize(
                        (img_height, img_width),
                        interpolation=T.InterpolationMode.BICUBIC,
                    )
                    cond_image = transform_hr(img_tensor.unsqueeze(0)).detach()
                else:
                    cond_image = img_tensor.unsqueeze(0).detach()

            if mask is None:
                cl_mask = None
            else:
                cl_mask = mask.unsqueeze(0).clone().detach()

            y_t, cond_image, img_tensor, mask = (
                y_t.unsqueeze(0).clone().detach(),
                cond_image.clone().detach() if cond_image is not None else None,
                img_tensor.unsqueeze(0).clone().detach(),
                cl_mask,
            )
            if mask is None:
                y0_tensor = None
            else:
                y0_tensor = img_tensor

            if opt.model_type == "palette":
                if "class" in model.denoise_fn.conditioning:
                    cls_tensor = (
                        torch.ones(1, dtype=torch.int64, device=device) * frame_cls
                    )
                else:
                    cls_tensor = None
            if ref is not None:
                ref_tensor = ref_tensor.unsqueeze(0)
            else:
                ref_tensor = None

            frame_data = {
                "index": sequence_count,
                "y_t": y_t,
                "cond_image": cond_image,
                "y0_tensor": y0_tensor,
                "mask": mask,
                "bbox_select": bbox_select,
                "img_orig": img_orig,
                "img_tensor": img_tensor,
                "bbox": bbox if bbox_in_path else None,
                "generated_bbox": generated_bbox,
                "has_bbox": bool(bbox_in_path),
                "ref_orig": ref_orig,
                "cls": int(frame_cls),
            }

            if prev_frame is None:
                prev_frame = frame_data
                if y0_tensor is not None:
                    B, C, H, W = y0_tensor.shape
                    bucket_idx = torch.randint(
                        0, num_buckets, (B,), device=y0_tensor.device
                    )
                    sigma_values = bucket_idx.float() / (num_buckets - 1) * max_sigma
                    sigma = sigma_values.view(B, 1, 1, 1)
                    eps_ctx = torch.randn(B, C, H, W, device=y0_tensor.device)
                    y0_noisy_first = y0_tensor + sigma * eps_ctx
                    if mask is not None:
                        noise_mask = torch.clamp(mask, 0, 1).to(dtype=y0_tensor.dtype)
                        noisy_first_for_save = (
                            y0_tensor * (1 - noise_mask) + y0_noisy_first * noise_mask
                        )
                    else:
                        noisy_first_for_save = y0_noisy_first
                    prev_frame["noisy_first"] = noisy_first_for_save.clone().detach()
                    prev_frame["noisy_first_sigma"] = (
                        sigma_values.detach().cpu().tolist()
                    )
                else:
                    y0_noisy_first = None
                continue

            if last_seq_half_y_t is None:
                input_cls_values = [prev_frame["cls"], frame_data["cls"]]
                if b2b_mask_condition:
                    if (
                        prev_frame["cond_image"] is None
                        or frame_data["cond_image"] is None
                        or prev_frame["mask"] is None
                        or frame_data["mask"] is None
                    ):
                        raise RuntimeError(
                            "B2B mask-as-channel streaming inference requires per-frame masks."
                        )

                    # First window should use the two real path frames. Only
                    # subsequent windows switch to generated-context + new frame.
                    y_t_batch = prepare_tensors([prev_frame["y_t"], frame_data["y_t"]])
                    y0_tensor_batch = prepare_tensors(
                        [prev_frame["y0_tensor"], frame_data["y0_tensor"]]
                    )
                    mask_batch = prepare_tensors(
                        [prev_frame["mask"], frame_data["mask"]]
                    )
                    cond_image_batch = prepare_tensors(
                        [prev_frame["cond_image"], frame_data["cond_image"]]
                    )
                    cond_output_prev = prev_frame["cond_image"]
                else:
                    y_t_batch = prepare_tensors([prev_frame["y_t"], frame_data["y_t"]])
                    y0_tensor_batch = prepare_tensors(
                        [prev_frame["y0_tensor"], frame_data["y0_tensor"]]
                    )
                    mask_batch = prepare_tensors(
                        [prev_frame["mask"], frame_data["mask"]]
                    )
                    if inputmix:
                        cond_image_batch = prepare_tensors(
                            [y0_noisy_first, frame_data["cond_image"]]
                        )
                        cond_output_prev = y0_noisy_first
                    else:
                        cond_image_batch = prepare_tensors(
                            [prev_frame["cond_image"], frame_data["cond_image"]]
                        )
                        cond_output_prev = prev_frame["cond_image"]
            else:
                input_y_t = last_seq_half_y_t + [frame_data["y_t"]]
                input_cond_image = last_seq_half_cond_image + [frame_data["cond_image"]]
                input_y0_tensor = last_seq_half_y0_tensor + [frame_data["y0_tensor"]]
                input_mask = last_seq_half_mask + [frame_data["mask"]]
                input_cls_values = last_seq_half_cls + [frame_data["cls"]]

                y_t_batch = prepare_tensors(input_y_t)
                cond_image_batch = prepare_tensors(input_cond_image)
                y0_tensor_batch = prepare_tensors(input_y0_tensor)
                mask_batch = prepare_tensors(input_mask)
                cond_output_prev = None

            with torch.no_grad():
                if opt.model_type == "palette":
                    out_tensor, _ = model.restoration(
                        y_cond=cond_image_batch,
                        y_t=y_t_batch,
                        y_0=y0_tensor_batch,
                        mask=mask_batch,
                        cls=cls_tensor,
                        ref=ref_tensor,
                        sample_num=2,
                        guidance_scale=alg_diffusion_guidance_scale,
                        ddim_num_steps=alg_palette_ddim_num_steps,
                        ddim_eta=alg_palette_ddim_eta,
                    )
                elif opt.model_type in ["cm", "cm_gan"]:
                    sampling_sigmas = (80.0, 24.4, 5.84, 0.9, 0.661)
                    out_tensor = model.restoration(
                        y_t_batch, cond_image_batch, sampling_sigmas, mask_batch
                    )
                elif opt.model_type == "b2b":
                    if b2b_mask_condition:
                        if y_t_batch.shape[2] != 3 or cond_image_batch.shape[2] != 1:
                            raise RuntimeError(
                                "Expected B2B streaming inputs to be RGB + 1-channel mask condition."
                            )
                    labels = prepare_label_tensor(input_cls_values, y_t_batch.device)
                    out_tensor = model.restoration(
                        y_t_batch,
                        cond_image_batch,
                        alg_b2b_denoise_timesteps,
                        mask_batch,
                        labels,
                    )

            out_tensor = out_tensor.squeeze(0)
            out_img_tensor_list_batch = []
            out_img_temp_list_batch = []
            for i in range(out_tensor.shape[0]):
                out_img = to_np(out_tensor[i : i + 1, :, :, :])
                out_img_tensor_list_batch.append(out_tensor[i : i + 1, :, :, :])
                out_img_temp_list_batch.append(out_img)

            y_t_temp_list = separate_tensors(y_t_batch)
            y0_tensor_temp_list = separate_tensors(y0_tensor_batch)
            mask_temp_list = separate_tensors(mask_batch)
            if b2b_mask_condition:
                generated_context = [
                    frame.clone().detach()
                    for frame in out_img_tensor_list_batch[-seq_half:]
                ]
                last_seq_half_y_t = generated_context
                last_seq_half_cond_image = [
                    torch.zeros_like(frame_data["cond_image"])
                    for _ in generated_context
                ]
                last_seq_half_y0_tensor = [
                    frame.clone().detach() for frame in generated_context
                ]
                last_seq_half_mask = [
                    torch.zeros_like(frame_data["mask"]) for _ in generated_context
                ]
            else:
                last_seq_half_y_t = y_t_temp_list[-seq_half:]
                last_seq_half_cond_image = out_img_tensor_list_batch[-seq_half:]
                last_seq_half_y0_tensor = y0_tensor_temp_list[-seq_half:]
                last_seq_half_mask = mask_temp_list[-seq_half:]
            last_seq_half_cls = input_cls_values[-seq_half:]

            if cond_output_prev is not None:
                write_frame(
                    prev_frame["index"],
                    out_img_temp_list_batch[0],
                    prev_frame,
                    cond_output_prev,
                )

            if b2b_mask_condition:
                cond_output_curr = frame_data["cond_image"]
            else:
                cond_output_curr = out_img_tensor_list_batch[-1]
            write_frame(
                frame_data["index"],
                out_img_temp_list_batch[-1],
                frame_data,
                cond_output_curr,
            )

            prev_frame = frame_data

            if frames_written > 0:
                generation_time = time.perf_counter() - start_time
                total_generation_time += generation_time
                avg_time = total_generation_time / num_generated_images
                frame_iter.set_postfix(avg_sec_per_image=avg_time)
    finally:
        if video_writer is not None:
            video_writer.release()

    if logger:
        logger.info(
            f"[it: %i/%i] - [4/%i] image written"
            % (iteration, nb_samples, PROGRESS_NUM_STEPS)
        )

    avg_generation_time_per_image = (
        total_generation_time / num_generated_images
        if num_generated_images > 0
        else 0.0
    )
    print(f"Average generation time per image: {avg_generation_time_per_image:.4f}s")

    return None, model, opt


def inference_logger(name):
    PROCESS_NAME = "gen_vid_diffusion"
    LOG_PATH = os.environ.get(
        "LOG_PATH", os.path.join(os.path.dirname(__file__), "../logs")
    )
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    #
    #    logging.basicConfig(
    #        level=logging.DEBUG,
    #        handlers=[
    #            logging.FileHandler(f"{LOG_PATH}/{name}.log", mode="w"),
    #            logging.StreamHandler(),
    #        ],
    #    )

    return logging.getLogger(f"inference %s %s" % (PROCESS_NAME, name))


def inference(args):
    PROGRESS_NUM_STEPS = 6
    logger = inference_logger(args.name)

    args.logger = logger

    if len(args.mask_delta_ratio[0]) == 1 and args.mask_delta_ratio[0][0] == 0.0:
        mask_delta = args.mask_delta
    else:
        mask_delta = args.mask_delta_ratio
    args.mask_delta = mask_delta

    args.write = True

    real_name = args.name

    # Load model once for the whole live run
    if not args.cpu:
        device = torch.device("cuda:" + str(args.gpuid))
    else:
        device = torch.device("cpu")

    lmodel, lopt = load_model(
        os.path.dirname(args.model_in_file),
        os.path.basename(args.model_in_file),
        device,
        args.sampling_steps,
        args.sampling_method,
        args.model_prior_321_backwardcompatibility,
    )
    args.lmodel = lmodel
    args.lopt = lopt

    for i in tqdm(range(args.nb_samples)):
        args.iteration = i + 1
        logger.info(f"[it: %i/%i] launch inference" % (args.iteration, args.nb_samples))
        args.name = real_name + "_" + str(i).zfill(len(str(args.nb_samples)))
        frame, lmodel, lopt = generate_streaming(**vars(args))

        args.lmodel = lmodel
        args.lopt = lopt

    logger.info(f"success - end of inference")


def extract_number(filename):
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else -1


def img2video(args, fps=10, ext=".avi", fourcc="MJPG"):
    image_folder = args.dir_out
    video_base_name = "video"

    for suffix in ["generated", "orig"]:
        # all files ending with this suffix
        files = [f for f in os.listdir(image_folder) if f.endswith(f"{suffix}.png")]
        if not files:
            continue

        # filenames start with a number: "1_generated.png" → 1
        files.sort(key=lambda x: int(x.split("_")[0]))
        files = files[::-1]

        # read first frame for size
        first_path = os.path.join(image_folder, files[0])
        first = cv2.imread(first_path)
        if first is None:
            print(f"Error reading {first_path}")
            continue
        H, W = first.shape[:2]

        # setup AVI writer
        out_name = f"{video_base_name}_{suffix}{ext}"
        out_path = os.path.join(image_folder, out_name)
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))

        for fname in files:
            frame = cv2.imread(os.path.join(image_folder, fname))
            if frame is None:
                continue
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))
            vw.write(frame)

        vw.release()
        cv2.destroyAllWindows()
        logging.info(f"Video created: {out_path}")
        print(f"Video created: {out_path}")


if __name__ == "__main__":
    args = InferenceDiffusionOptions().parse(save_config=False)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Put the generated frame on the left side of the original frame in the AVI output",
    )
    extra_args, _ = parser.parse_known_args()
    args.compare = extra_args.compare

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out, exist_ok=True)
        print(f"[info] Created output directory: {args.dir_out}")
    inference(args)
