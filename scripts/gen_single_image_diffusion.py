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

    # if opt.G_nblocks == 9:
    #    warnings.warn(
    #        f"G_nblocks default value {opt.G_nblocks} is too high for palette model, 2 will be used instead."
    #    )
    #    opt.G_nblocks = 2

    if opt.data_online_creation_mask_random_offset_A != [0.0]:
        warnings.warn(
            f"disabling data_online_creation_mask_random_offset_A in inference mode"
        )
        opt.data_online_creation_mask_random_offset_A = [0.0]

    opt.model_prior_321_backwardcompatibility = model_prior_321_backwardcompatibility
    if opt.model_type in ["cm", "cm_gan", "sc", "b2b"]:
        opt.alg_palette_sampling_method = sampling_method
        opt.alg_diffusion_cond_embed_dim = 256
    model = diffusion_networks.define_G(**vars(opt))
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
        # model.denoise_fn.model.beta_schedule["test"]["schedule"] = "quad"
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


def generate(
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
    alg_sc_denoise_inferstep,
    alg_b2b_denoise_inferstep,
    data_refined_mask,
    min_crop_bbox_ratio,
    alg_palette_ddim_num_steps,
    alg_palette_ddim_eta,
    model_prior_321_backwardcompatibility,
    logger,
    iteration,
    nb_samples,
    **unused_options,
):
    PROGRESS_NUM_STEPS = 4
    # seed
    if seed >= 0:
        torch.manual_seed(seed)

    if not cpu:
        device = torch.device("cuda:" + str(gpuid))
    else:
        device = torch.device("cpu")

    # loading model
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

    # Load image

    # reading image
    if opt.data_image_bits > 8:
        img = Image.open(img_in)  # we use PIL
        img_orig = img.copy()
        local_img_width, local_img_height = img.size
    else:
        img = cv2.imread(img_in)
        img_orig = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        local_img_width, local_img_height = img.shape[:2]

    # reading the mask
    mask = None
    if mask_in:
        mask = cv2.imread(mask_in, 0)

    # reading reference image
    ref = None
    if ref_in:
        ref = cv2.imread(ref_in)
        ref_orig = ref.copy()
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    bboxes = []
    if bbox_in:
        # mask = np.zeros(img.shape[:2], dtype=np.uint8)
        with open(bbox_in, "r") as bboxf:
            for line in bboxf:
                elts = line.rstrip().split()
                bboxes.append([int(elts[1]), int(elts[2]), int(elts[3]), int(elts[4])])
                if conditioning:
                    if cls <= 0:
                        cls = int(elts[0])
                else:
                    cls = 1

        if bbox_ref_id == -1:
            # sample a bbox here since we are calling crop_image multiple times
            bbox_idx = random.choice(range(len(bboxes)))
        else:
            bbox_idx = bbox_ref_id

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

            # TODO: unused?
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
            bbox_path=bbox_in,
            mask_delta=mask_delta,  # =opt.data_online_creation_mask_delta_A,
            mask_random_offset=opt.data_online_creation_mask_random_offset_A,
            crop_delta=0,
            mask_square=mask_square,  # opt.data_online_creation_mask_square_A,
            crop_dim=opt.data_online_creation_crop_size_A,  # we use the average crop_dim
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
            bbox_path=bbox_in,
            mask_delta=mask_delta,  # opt.data_online_creation_mask_delta_A,
            mask_random_offset=opt.data_online_creation_mask_random_offset_A,
            crop_delta=0,
            mask_square=mask_square,  # opt.data_online_creation_mask_square_A,
            crop_dim=opt.data_online_creation_crop_size_A,  # we use the average crop_dim
            output_dim=opt.data_load_size,
            context_pixels=opt.data_online_context_pixels,
            load_size=opt.data_online_creation_load_size_A,
            crop_coordinates=crop_coordinates,
            crop_center=True,
            bbox_ref_id=bbox_idx,
            override_class=cls,
        )

        x_crop, y_crop, crop_size = crop_coordinates

        bbox = bboxes[bbox_idx]

        bbox_select = bbox.copy()

        if len(mask_delta) == 1:
            index_cls = 0
        else:
            index_cls = int(cls) - 1

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
            )  # (xmax - xmin) - (ymax - ymin)
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

        # clamp bbox to image size
        bbox_select[0] = max(0, bbox_select[0])
        bbox_select[1] = max(0, bbox_select[1])
        bbox_select[2] = min(img_orig.shape[1], bbox_select[2])
        bbox_select[3] = min(img_orig.shape[0], bbox_select[3])

        img, mask = np.array(img), np.array(mask)

    if img_width > 0 and img_height > 0:
        if img_height != local_img_height or img_width != local_img_width:
            if opt.data_image_bits > 8:
                print(
                    "Requested image size differs from training crop size, resizing is not supported for images with more than 8 bits per channel"
                )
                exit(1)
            img = cv2.resize(img, (img_width, img_height))
            if mask is not None:
                mask = cv2.resize(mask, (img_width, img_height))
            if ref is not None:
                ref = cv2.resize(ref, (img_width, img_height))

    if logger:
        logger.info(
            f"[it: %i/%i] - [2/%i] image loaded"
            % (iteration, nb_samples, PROGRESS_NUM_STEPS)
        )

    # insert cond image into original image
    generated_bbox = None
    if cond_in:
        generated_bbox = bbox
        # fill the mask with cond image
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
                (x0 + x * bbox_w / cond.shape[1], y0 + y * bbox_h / cond.shape[0])
                for x, y in generated_bbox
            ]
            # bbox inside original image
            real_width = min(img_orig.shape[1], bbox_select[2] - bbox_select[0])
            real_height = min(img_orig.shape[0], bbox_select[3] - bbox_select[1])
            generated_bbox = [
                (
                    bbox_select[0] + x * real_width / img.shape[1],
                    bbox_select[1] + y * real_height / img.shape[0],
                )
                for x, y in generated_bbox
            ]
            # round & flatten
            generated_bbox = list(map(round, generated_bbox[0] + generated_bbox[1]))

        cond = cv2.resize(cond, (bbox_w, bbox_h), interpolation=cv2.INTER_CUBIC)
        img[y0:y1, x0:x1] = cond

    # preprocessing to torch
    totensor = transforms.ToTensor()
    if opt.data_image_bits > 8:
        tranlist = [totensor, torchvision.transforms.v2.ToDtype(torch.float32)]
        bit_scaling = 2**opt.data_image_bits - 1
        tranlist += [transforms.Lambda(lambda img: img * (1 / float(bit_scaling)))]
        tranlist += [
            transforms.Normalize((0.5,), (0.5,))
        ]  # XXX: > 8bit, mono canal only for now
    else:
        tranlist = [
            totensor,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #    resize,
        ]

    tran = transforms.Compose(tranlist)
    img_tensor = tran(img).clone().detach()

    if mask is not None:
        mask = torch.from_numpy(np.array(mask, dtype=np.int64)).unsqueeze(0)
        """if crop_width > 0 and crop_height > 0:
        mask = resize(mask).clone().detach()"""
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
                # load the previous frame
                previous_frame = cv2.imread(previous_frame)

            previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB)
            previous_frame = previous_frame[
                bbox_select[1] : bbox_select[3], bbox_select[0] : bbox_select[2]
            ]
            previous_frame = cv2.resize(
                previous_frame, (opt.data_load_size, opt.data_load_size)
            )
            previous_frame = tran(previous_frame)
            previous_frame = previous_frame.to(device).clone().detach().unsqueeze(0)

            cond_image = previous_frame
        else:
            cond_image = -1 * torch.ones_like(y_t.unsqueeze(0), device=y_t.device)
    elif opt.alg_diffusion_cond_image_creation == "y_t":
        if opt.model_type == "palette":
            cond_image = y_t.unsqueeze(0)
        else:
            cond_image = None
    elif opt.alg_diffusion_cond_image_creation == "sketch":
        cond_image = fill_img_with_sketch(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_diffusion_cond_image_creation == "canny":
        clamp = torch.clamp(mask, 0, 1)
        if cond_in:
            # apply canny to the bbox to avoid edges around cond image
            img_tensor_canny = img_tensor[:, y0:y1, x0:x1]
            mask_canny = mask[:, y0:y1, x0:x1]
        else:
            img_tensor_canny = img_tensor
            mask_canny = mask
        canny = fill_img_with_canny(
            img_tensor_canny.unsqueeze(0),
            mask_canny.unsqueeze(0),
            low_threshold=alg_diffusion_sketch_canny_thresholds[0],
            high_threshold=alg_diffusion_sketch_canny_thresholds[1],
            low_threshold_random=-1,
            high_threshold_random=-1,
            select_canny=[1],
        )
        if cond_in:
            # restore background
            cond_image = img_tensor.unsqueeze(0).clone()
            cond_image[:, :, y0:y1, x0:x1] = canny
        else:
            cond_image = canny
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
        cond_image = fill_img_with_hed(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_diffusion_cond_image_creation == "hough":
        cond_image = fill_img_with_hough(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_diffusion_cond_image_creation == "depth":
        cond_image = fill_img_with_depth(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_diffusion_cond_image_creation == "low_res":
        if alg_diffusion_super_resolution_downsample:
            data_crop_size_low_res = int(
                opt.data_crop_size / opt.alg_diffusion_super_resolution_scale
            )
            transform_lr = T.Resize((data_crop_size_low_res, data_crop_size_low_res))
            cond_image = transform_lr(img_tensor.unsqueeze(0)).detach()
        else:
            cond_image = img_tensor.unsqueeze(0).clone().detach()
        transform_hr = T.Resize((opt.data_crop_size, opt.data_crop_size))
        cond_image = transform_hr(cond_image).detach()
    elif opt.alg_diffusion_cond_image_creation == "pix2pix":
        # use same interpolation as get_transform
        if (img_height > 0 and img_height != opt.data_crop_size) or (
            img_width > 0 and img_width != opt.data_crop_size
        ):
            transform_hr = T.Resize(
                (img_height, img_width), interpolation=T.InterpolationMode.BICUBIC
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
    if mask == None:
        y0_tensor = None
    else:
        y0_tensor = img_tensor

    if opt.model_type == "palette":
        if "class" in model.denoise_fn.conditioning:
            cls_tensor = torch.ones(1, dtype=torch.int64, device=device) * cls
        else:
            cls_tensor = None
    if ref is not None:
        ref_tensor = ref_tensor.unsqueeze(0)
    else:
        ref_tensor = None

    # run through model
    with torch.no_grad():
        if opt.model_type == "palette":
            out_tensor, visu = model.restoration(
                y_cond=cond_image,
                y_t=y_t,
                y_0=y0_tensor,
                mask=mask,
                cls=cls_tensor,
                ref=ref_tensor,
                sample_num=2,
                guidance_scale=alg_diffusion_guidance_scale,
                ddim_num_steps=alg_palette_ddim_num_steps,
                ddim_eta=alg_palette_ddim_eta,
            )
        elif opt.model_type == "cm" or opt.model_type == "cm_gan":
            sampling_sigmas = (80.0, 24.4, 5.84, 0.9, 0.661)

            out_tensor = model.restoration(y_t, cond_image, sampling_sigmas, mask)

        elif opt.model_type == "sc":
            print(" sc restoration ", alg_sc_denoise_inferstep)
            out_tensor = model.restoration(
                y_t, cond_image, alg_sc_denoise_inferstep, mask
            )

        elif opt.model_type == "b2b":
            B = y_t.size(0)
            labels = torch.zeros(B, device=y_t.device, dtype=torch.long)
            out_tensor = model.restoration(
                y_t, cond_image, alg_b2b_denoise_inferstep, mask, labels
            )

        # XXX: !=8bit images are converted to 8bit RGB for now
        out_img = to_np(
            out_tensor
        )  # out_img = out_img.detach().data.cpu().float().numpy()[0]

    if logger:
        logger.info(
            f"[it: %i/%i] - [3/%i] processing completed"
            % (iteration, nb_samples, PROGRESS_NUM_STEPS)
        )

    """ post-processing
    
    out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)"""

    if bbox_in:
        out_img_resized = cv2.resize(
            out_img,
            (
                min(img_orig.shape[1], bbox_select[2] - bbox_select[0]),
                min(img_orig.shape[0], bbox_select[3] - bbox_select[1]),
            ),
        )

        out_img_real_size = img_orig.copy()
    else:
        out_img_resized = out_img
        out_img_real_size = img_orig.copy()

    # fill out crop into original image
    if bbox_in:
        out_img_real_size[
            bbox_select[1] : bbox_select[3], bbox_select[0] : bbox_select[2]
        ] = out_img_resized

    if cond_image is not None:
        cond_img = to_np(cond_image)

    if write:
        if opt.data_image_bits > 8:
            img_np = to_np(img_tensor)  # comes from PIL
            cv2.imwrite(os.path.join(dir_out, name + "_orig.png"), img_np)
            if cond_image is not None:
                cv2.imwrite(os.path.join(dir_out, name + "_cond.png"), cond_img)
            cv2.imwrite(os.path.join(dir_out, name + "_generated.png"), out_img_resized)
        else:
            cv2.imwrite(os.path.join(dir_out, name + "_orig.png"), img_orig)
            if cond_image is not None:
                cv2.imwrite(os.path.join(dir_out, name + "_cond.png"), cond_img)
            cv2.imwrite(
                os.path.join(dir_out, name + "_generated.png"), out_img_real_size
            )
            cv2.imwrite(os.path.join(dir_out, name + "_y_t.png"), to_np(y_t))
            if mask is not None:
                cv2.imwrite(os.path.join(dir_out, name + "_y_0.png"), to_np(img_tensor))
                cv2.imwrite(
                    os.path.join(dir_out, name + "_generated_crop.png"), out_img
                )
                cv2.imwrite(os.path.join(dir_out, name + "_mask.png"), to_np(mask))
            if ref is not None:
                cv2.imwrite(os.path.join(dir_out, name + "_ref_orig.png"), ref_orig)
            if cond_in:
                # crop before cond image
                orig_crop = img_orig[
                    bbox_select[1] : bbox_select[3], bbox_select[0] : bbox_select[2]
                ]
                cv2.imwrite(os.path.join(dir_out, name + "_orig_crop.png"), orig_crop)
            if bbox_in:
                with open(os.path.join(dir_out, name + "_orig_bbox.json"), "w") as out:
                    out.write(json.dumps(bbox))
            if generated_bbox:
                with open(
                    os.path.join(dir_out, name + "_generated_bbox.json"), "w"
                ) as out:
                    out.write(json.dumps(generated_bbox))

        print("Successfully generated image ", name)

    if logger:
        logger.info(
            f"[it: %i/%i] - [4/%i] image written"
            % (iteration, nb_samples, PROGRESS_NUM_STEPS)
        )

    return out_img_real_size, model, opt


def inference_logger(name):
    PROCESS_NAME = "gen_single_image_diffusion"
    LOG_PATH = os.environ.get(
        "LOG_PATH", os.path.join(os.path.dirname(__file__), "../logs")
    )
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(f"{LOG_PATH}/{name}.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

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

    args.lmodel = None
    args.lopt = None

    for i in tqdm(range(args.nb_samples)):
        args.iteration = i + 1
        logger.info(f"[it: %i/%i] launch inference" % (args.iteration, args.nb_samples))
        args.name = real_name + "_" + str(i).zfill(len(str(args.nb_samples)))
        frame, lmodel, lopt = generate(**vars(args))
        args.lmodel = lmodel
        args.lopt = lopt

    logger.info(f"success - end of inference")


if __name__ == "__main__":
    args = InferenceDiffusionOptions().parse(save_config=False)
    inference(args)
