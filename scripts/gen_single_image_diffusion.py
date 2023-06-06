import argparse
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


def load_model(modelpath, model_in_file, device, sampling_steps, sampling_method):
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

    model = diffusion_networks.define_G(**vars(opt))
    model.eval()
    model.load_state_dict(torch.load(modelpath + "/" + model_in_file))

    # sampling steps
    if sampling_steps > 0:
        model.denoise_fn.beta_schedule["test"]["n_timestep"] = sampling_steps
        set_new_noise_schedule(model.denoise_fn, "test")

    model.set_new_sampling_method(sampling_method)

    model = model.to(device)
    return model, opt


def to_np(img):
    img = img.detach().data.cpu().float().numpy()[0]
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


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
    data_refined_mask,
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
    )

    if alg_palette_cond_image_creation is not None:
        opt.alg_palette_cond_image_creation = alg_palette_cond_image_creation

    conditioning = opt.alg_palette_conditioning
    print("conditioning=", conditioning)

    if len(opt.data_online_creation_mask_delta_A) == 1:
        opt.data_online_creation_mask_delta_A.append(
            opt.data_online_creation_mask_delta_A[0]
        )

    if len(mask_delta) == 1:
        mask_delta.append(mask_delta[0])

    if opt.data_online_creation_mask_square_A:
        mask_square = True

    mask_delta[0] += opt.data_online_creation_mask_delta_A[0]
    mask_delta[1] += opt.data_online_creation_mask_delta_A[1]

    # Load image

    # reading image
    img = cv2.imread(img_in)
    img_orig = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # reading the mask
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

        # insert cond image into original image
        if cond_in:
            x0, y0, x1, y1 = bbox
            w = x1 - x0
            h = y1 - y0
            cond = cv2.imread(cond_in)
            cond = cv2.resize(cond, (w, h), interpolation=cv2.INTER_CUBIC)
            orig = img_orig.copy()
            orig[y0:y1, x0:x1] = cond
            img_in = tempfile.NamedTemporaryFile(suffix=".png").name
            cv2.imwrite(img_in, orig)

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

        bbox_select[0] -= mask_delta[0]
        bbox_select[1] -= mask_delta[1]
        bbox_select[2] += mask_delta[0]
        bbox_select[3] += mask_delta[1]

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
    else:
        mask = None

    if img_width > 0 and img_height > 0:
        img = cv2.resize(img, (img_width, img_height))
        if mask is not None:
            mask = cv2.resize(mask, (img_width, img_height))

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
        cond_image = fill_img_with_canny(
            img_tensor.unsqueeze(0),
            mask.unsqueeze(0),
            low_threshold=alg_palette_sketch_canny_thresholds[0],
            high_threshold=alg_palette_sketch_canny_thresholds[1],
            low_threshold_random=-1,
            high_threshold_random=-1,
        )
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

    if "class" in model.conditioning:
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
        help="sobel threshold in % of gradient magintude",
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

    args = options.parse()

    args.write = True

    real_name = args.name

    for i in tqdm(range(args.nb_samples)):
        args.name = real_name + "_" + str(i).zfill(len(str(args.nb_samples)))
        generate(**vars(args))
