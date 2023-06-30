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

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append("../")
from diffusion_options import DiffusionOptions
from segment_anything import SamPredictor

from data.online_creation import crop_image, fill_mask_with_color, fill_mask_with_random
from models import diffusion_networks
from models.modules.diffusion_utils import set_new_noise_schedule
from models.modules.sam.sam_inference import (
    compute_mask_with_sam,
    load_sam_weight,
    predict_sam_mask,
)
from models.modules.utils import download_sam_weight
from options.train_options import TrainOptions
from util.mask_generation import (
    fill_img_with_canny,
    fill_img_with_depth,
    fill_img_with_hed,
    fill_img_with_hough,
    fill_img_with_sam,
    fill_img_with_sketch,
)


def load_model(
    modelpath,
    model_in_file,
    device,
    sampling_steps,
    sampling_method,
    model_prior_321_backwardcompatibility,
):
    train_json_path = modelpath + "/train_config.json"
    with open(train_json_path, "r") as jsonf:
        train_json = json.load(jsonf)

    opt = TrainOptions().parse_json(train_json)
    opt.jg_dir = "../"

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
    model = diffusion_networks.define_G(**vars(opt))
    model.eval()

    # handle old models
    weights = torch.load(modelpath + "/" + model_in_file)
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

    model.set_new_sampling_method(sampling_method)

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
    cpu,
    gpuid,
    sampling_steps,
    img_in,
    mask_in,
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
    alg_palette_cond_image_creation,
    alg_palette_sketch_canny_thresholds,
    cls,
    alg_palette_super_resolution_downsample,
    alg_palette_guidance_scale,
    data_refined_mask,
    min_crop_bbox_ratio,
    ddim_num_steps,
    ddim_eta,
    **unused_options,
):
    # seed
    if seed >= 0:
        torch.manual_seed(seed)

    # loading model
    modelpath = model_in_file.replace(os.path.basename(model_in_file), "")

    if not cpu:
        device = torch.device("cuda:" + str(gpuid))
    else:
        device = torch.device("cpu")
    model, opt = load_model(
        modelpath,
        os.path.basename(model_in_file),
        device,
        sampling_steps,
        sampling_method,
        args.model_prior_321_backwardcompatibility,
    )

    if alg_palette_cond_image_creation is not None:
        opt.alg_palette_cond_image_creation = alg_palette_cond_image_creation

    conditioning = opt.alg_palette_conditioning

    for i, delta_values in enumerate(mask_delta):
        if len(delta_values) == 1:
            mask_delta[i].append(delta_values[0])

    # Load image

    # reading image
    img = cv2.imread(img_in)
    img_orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # reading the mask
    mask = None
    if mask_in:
        mask = cv2.imread(mask_in, 0)

    bboxes = []
    if bbox_in:
        # mask = np.zeros(img.shape[:2], dtype=np.uint8)
        with open(bbox_in, "r") as bboxf:
            for line in bboxf:
                elts = line.rstrip().split()
                bboxes.append([int(elts[1]), int(elts[2]), int(elts[3]), int(elts[4])])
                if conditioning:
                    if args.cls:
                        cls = args.cls
                    else:
                        cls = elts[0]
                        print("generating with class=", cls)
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

        img, mask, ref_bbox = crop_image(
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

        img, mask = np.array(img), np.array(mask)

    if img_width > 0 and img_height > 0:
        img = cv2.resize(img, (img_width, img_height))
        if mask is not None:
            mask = cv2.resize(mask, (img_width, img_height))

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
            args.cond_rotation,
            args.cond_persp_horizontal,
            args.cond_persp_vertical,
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

        # add 1 pixel margin for sketches
        cond = cv2.resize(cond, (bbox_w - 2, bbox_h - 2), interpolation=cv2.INTER_CUBIC)
        cond = np.pad(cond, [(1, 1), (1, 1), (0, 0)])
        img[y0:y1, x0:x1] = cond

    # preprocessing to torch
    totensor = transforms.ToTensor()
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

    if not cpu:
        img_tensor = img_tensor.to(device).clone().detach()
        if mask is not None:
            mask = mask.to(device).clone().detach()

    if mask is not None:
        if data_refined_mask:
            opt.f_s_weight_sam = "../" + opt.f_s_weight_sam
            if not os.path.exists(opt.f_s_weight_sam):
                download_sam_weight(path=opt.f_s_weight_sam)
            sam_model, _ = load_sam_weight(model_path=opt.f_s_weight_sam)
            sam_model = sam_model.to(device)
            mask = compute_mask_with_sam(
                img_tensor, mask, sam_model, device, batched=False
            ).unsqueeze(0)

        if opt.data_online_creation_rand_mask_A:
            y_t = fill_mask_with_random(
                img_tensor.clone().detach(), mask.clone().detach(), -1
            )
        elif opt.data_online_creation_color_mask_A:
            y_t = fill_mask_with_color(
                img_tensor.clone().detach(), mask.clone().detach(), {}
            )
    else:
        y_t = img_tensor.clone().detach()

    if opt.alg_palette_cond_image_creation == "previous_frame":
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
    elif opt.alg_palette_cond_image_creation == "y_t":
        cond_image = y_t.unsqueeze(0)
    elif opt.alg_palette_cond_image_creation == "sketch":
        cond_image = fill_img_with_sketch(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_palette_cond_image_creation == "canny":
        clamp = torch.clamp(mask, 0, 1)
        if cond_in:
            # mask the background to avoid canny edges around cond image
            img_tensor_canny = clamp * img_tensor + clamp - 1
        else:
            img_tensor_canny = img_tensor
        cond_image = fill_img_with_canny(
            img_tensor_canny.unsqueeze(0),
            mask.unsqueeze(0),
            low_threshold=alg_palette_sketch_canny_thresholds[0],
            high_threshold=alg_palette_sketch_canny_thresholds[1],
            low_threshold_random=-1,
            high_threshold_random=-1,
        )
        if cond_in:
            # restore background
            cond_image = cond_image * clamp + img_tensor * (1 - clamp)
    elif opt.alg_palette_cond_image_creation == "sam":
        opt.f_s_weight_sam = "../" + opt.f_s_weight_sam
        if not os.path.exists(opt.f_s_weight_sam):
            download_sam_weight(opt.f_s_weight_sam)
        sam, _ = load_sam_weight(opt.f_s_weight_sam)
        sam = sam.to(device)
        cond_image = fill_img_with_sam(
            img_tensor.unsqueeze(0), mask.unsqueeze(0), sam, opt
        )
    elif opt.alg_palette_cond_image_creation == "hed":
        cond_image = fill_img_with_hed(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_palette_cond_image_creation == "hough":
        cond_image = fill_img_with_hough(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_palette_cond_image_creation == "depth":
        cond_image = fill_img_with_depth(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_palette_cond_image_creation == "low_res":
        if alg_palette_super_resolution_downsample:
            data_crop_size_low_res = int(
                opt.data_crop_size / opt.alg_palette_super_resolution_scale
            )
            transform_lr = T.Resize((data_crop_size_low_res, data_crop_size_low_res))
            cond_image = transform_lr(img_tensor.unsqueeze(0)).detach()
        else:
            cond_image = img_tensor.unsqueeze(0).clone().detach()
        transform_hr = T.Resize((opt.data_crop_size, opt.data_crop_size))
        cond_image = transform_hr(cond_image).detach()

    # run through model
    if mask is None:
        cl_mask = None

    else:
        cl_mask = mask.unsqueeze(0).clone().detach()
    y_t, cond_image, img_tensor, mask = (
        y_t.unsqueeze(0).clone().detach(),
        cond_image.clone().detach(),
        img_tensor.unsqueeze(0).clone().detach(),
        cl_mask,
    )
    if mask == None:
        img_tensor = None

    if "class" in model.denoise_fn.conditioning:
        cls_tensor = torch.ones(1, dtype=torch.int64, device=device) * cls
    else:
        cls_tensor = None

    with torch.no_grad():
        out_tensor, visu = model.restoration(
            y_cond=cond_image,
            y_t=y_t,
            y_0=img_tensor,
            mask=mask,
            cls=cls_tensor,
            sample_num=2,
            guidance_scale=alg_palette_guidance_scale,
            ddim_num_steps=ddim_num_steps,
            ddim_eta=ddim_eta,
        )
        out_img = to_np(
            out_tensor
        )  # out_img = out_img.detach().data.cpu().float().numpy()[0]

    """ post-processing
    
    out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)"""

    if img_width > 0 or img_height > 0 or crop_width > 0 or crop_height > 0:
        # img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)

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
            out_img_real_size = out_img

    # fill out crop into original image
    if bbox_in:
        out_img_real_size[
            bbox_select[1] : bbox_select[3], bbox_select[0] : bbox_select[2]
        ] = out_img_resized

    cond_img = to_np(cond_image)

    if write:
        cv2.imwrite(os.path.join(dir_out, name + "_orig.png"), img_orig)
        cv2.imwrite(os.path.join(dir_out, name + "_cond.png"), cond_img)
        cv2.imwrite(os.path.join(dir_out, name + "_generated.png"), out_img_real_size)
        cv2.imwrite(os.path.join(dir_out, name + "_y_t.png"), to_np(y_t))
        if mask is not None:
            cv2.imwrite(os.path.join(dir_out, name + "_y_0.png"), to_np(img_tensor))
            cv2.imwrite(os.path.join(dir_out, name + "_generated_crop.png"), out_img)
            cv2.imwrite(os.path.join(dir_out, name + "_mask.png"), to_np(mask))
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
            with open(os.path.join(dir_out, name + "_generated_bbox.json"), "w") as out:
                out.write(json.dumps(generated_bbox))

        print("Successfully generated image ", name)

    return out_img_real_size


if __name__ == "__main__":
    options = DiffusionOptions()

    options.parser.add_argument("--img-in", help="image to transform", required=True)
    options.parser.add_argument(
        "--previous-frame", help="image to transform", default=None
    )
    options.parser.add_argument(
        "--mask-in", help="mask used for image transformation", required=False
    )
    options.parser.add_argument("--bbox-in", help="bbox file used for masking")

    options.parser.add_argument(
        "--nb_samples", help="nb of samples generated", type=int, default=1
    )
    options.parser.add_argument(
        "--bbox_ref_id", help="bbox id to use", type=int, default=-1
    )
    options.parser.add_argument("--cond-in", help="conditionning image to use")
    options.parser.add_argument("--cond_keep_ratio", action="store_true")
    options.parser.add_argument("--cond_rotation", type=float, default=0)
    options.parser.add_argument("--cond_persp_horizontal", type=float, default=0)
    options.parser.add_argument("--cond_persp_vertical", type=float, default=0)
    options.parser.add_argument(
        "--alg_palette_cond_image_creation",
        type=str,
        choices=[
            "y_t",
            "previous_frame",
            "sketch",
            "canny",
            "depth",
            "hed",
            "hough",
            "low_res",
            "sam",
        ],
        help="how cond_image is created",
    )
    options.parser.add_argument(
        "--alg_palette_sketch_canny_thresholds",
        type=int,
        nargs="+",
        default=[0, 255 * 3],
        help="Canny thresholds",
    )
    options.parser.add_argument(
        "--alg_palette_super_resolution_downsample",
        action="store_true",
        help="whether to downsample the image for super resolution",
    )

    options.parser.add_argument(
        "--alg_palette_sam_use_gaussian_filter",
        action="store_true",
        default=False,
        help="whether to apply a gaussian blur to each SAM masks",
    )

    options.parser.add_argument(
        "--alg_palette_sam_no_sobel_filter",
        action="store_false",
        default=True,
        help="whether to not use a Sobel filter on each SAM masks",
    )

    options.parser.add_argument(
        "--alg_palette_sam_no_output_binary_sam",
        action="store_false",
        default=True,
        help="whether to not output binary sketch before Canny",
    )

    options.parser.add_argument(
        "--alg_palette_sam_redundancy_threshold",
        type=float,
        default=0.62,
        help="redundancy threshold above which redundant masks are not kept",
    )

    options.parser.add_argument(
        "--alg_palette_sam_sobel_threshold",
        type=float,
        default=0.7,
        help="sobel threshold in %% of gradient magintude",
    )

    options.parser.add_argument(
        "--alg_palette_sam_final_canny",
        action="store_true",
        default=False,
        help="whether to perform a Canny edge detection on sam sketch to soften the edges",
    )

    options.parser.add_argument(
        "--alg_palette_sam_min_mask_area",
        type=float,
        default=0.001,
        help="minimum area in proportion of image size for a mask to be kept",
    )

    options.parser.add_argument(
        "--alg_palette_sam_max_mask_area",
        type=float,
        default=0.99,
        help="maximum area in proportion of image size for a mask to be kept",
    )

    options.parser.add_argument(
        "--alg_palette_sam_points_per_side",
        type=int,
        default=16,
        help="number of points per side of image to prompt SAM with (# of prompted points will be points_per_side**2)",
    )

    options.parser.add_argument(
        "--alg_palette_sam_no_sample_points_in_ellipse",
        action="store_false",
        default=True,
        help="whether to not sample the points inside an ellipse to avoid the corners of the image",
    )

    options.parser.add_argument(
        "--alg_palette_sam_crop_delta",
        type=int,
        default=True,
        help="extend crop's width and height by 2*crop_delta before computing masks",
    )

    options.parser.add_argument(
        "--alg_palette_guidance_scale",
        type=float,
        default=0.0,  # literature value: 0.2
        help="scale for classifier-free guidance, default is conditional DDPM only",
    )

    options.parser.add_argument(
        "--f_s_weight_sam",
        type=str,
        default="models/configs/sam/pretrain/sam_vit_b_01ec64.pth",
        help="path to sam weight for f_s, e.g. models/configs/sam/pretrain/sam_vit_b_01ec64.pth",
    )

    options.parser.add_argument(
        "--data_refined_mask",
        action="store_true",
        help="whether to use refined mask with sam",
    )

    options.parser.add_argument(
        "--min_crop_bbox_ratio",
        type=float,
        help="minimum crop/bbox ratio, allows to add context when bbox is larger than crop",
    )

    options.parser.add_argument(
        "--model_prior_321_backwardcompatibility",
        action="store_true",
        help="whether to load models from previous version of JG.",
    )

    args = options.parse()

    if len(args.mask_delta_ratio[0]) == 1 and args.mask_delta_ratio[0][0] == 0.0:
        mask_delta = args.mask_delta
    else:
        mask_delta = args.mask_delta_ratio
    args.mask_delta = mask_delta

    args.write = True

    real_name = args.name

    for i in tqdm(range(args.nb_samples)):
        args.name = real_name + "_" + str(i).zfill(len(str(args.nb_samples)))
        generate(**vars(args))
