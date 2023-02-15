import sys
import os
import json
import random
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import argparse
import warnings
import math
from tqdm import tqdm

sys.path.append("../")
from models import diffusion_networks
from options.train_options import TrainOptions
from data.online_creation import (
    fill_mask_with_random,
    fill_mask_with_color,
    crop_image,
    create_mask,
)
from models.modules.diffusion_utils import set_new_noise_schedule
from util.mask_generation import fill_img_with_sketch, fill_img_with_edges
from diffusion_options import DiffusionOptions


def load_model(modelpath, model_in_file, device, sampling_steps):
    train_json_path = modelpath + "/train_config.json"
    with open(train_json_path, "r") as jsonf:
        train_json = json.load(jsonf)

    opt = TrainOptions().parse_json(train_json)
    opt.jg_dir = "../"

    if opt.G_nblocks == 9:
        warnings.warn(
            f"G_nblocks default value {opt.G_nblocks} is too high for palette model, 2 will be used instead."
        )
        opt.G_nblocks = 2

    model = diffusion_networks.define_G(**vars(opt))
    # model.eval()
    model.load_state_dict(torch.load(modelpath + "/" + model_in_file), strict=False)

    # sampling steps
    if sampling_steps > 0:
        model.denoise_fn.beta_schedule["test"]["n_timestep"] = sampling_steps
        set_new_noise_schedule(model.denoise_fn, "test")

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
    bbox_width_factor,
    bbox_height_factor,
    crop_width,
    crop_height,
    img_width,
    img_height,
    dir_out,
    write,
    previous_frame,
    previous_frame_bbox_in,
    name,
    mask_delta,
    mask_square,
    reconstruction_guidance,
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
        modelpath, os.path.basename(model_in_file), device, sampling_steps
    )

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

        if crop_width or crop_height > 0:
            hc_width = int(crop_width / 2)
            hc_height = int(crop_height / 2)
            # select one bbox and crop around it
            bbox_orig = random.choice(bboxes)
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

            """
            mask = np.zeros(
                (bbox_select[3] - bbox_select[1], bbox_select[2] - bbox_select[0]),
                dtype=np.uint8,
            )
            mask[
                hc_height : hc_height + (bbox_orig[3] - bbox_orig[1]),
                hc_width : hc_width + bbox_orig[2] - bbox_orig[0],
            ] = np.full((bbox_orig[3] - bbox_orig[1], bbox_orig[2] - bbox_orig[0]), 1)
            img_orig = img.copy()
            img = img[
                bbox_select[1] : bbox_select[3], bbox_select[0] : bbox_select[2]
            ]  # cropped img"""

        else:
            for bbox in bboxes:
                mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = np.full(
                    (bbox[3] - bbox[1], bbox[2] - bbox[0]), 1
                )  # ymin:ymax, xmin:xmax, ymax-ymin, xmax-xmin

        crop_coordinates = crop_image(
            img_path=img_in,
            bbox_path=bbox_in,
            mask_delta=mask_delta,  # =opt.data_online_creation_mask_delta_A,
            crop_delta=0,
            mask_square=mask_square,  # opt.data_online_creation_mask_square_A,
            crop_dim=opt.data_online_creation_crop_size_A,  # we use the average crop_dim
            output_dim=opt.data_load_size,
            context_pixels=opt.data_online_context_pixels,
            load_size=opt.data_online_creation_load_size_A,
            get_crop_coordinates=True,
            crop_center=True,
        )

        img, mask = crop_image(
            img_path=img_in,
            bbox_path=bbox_in,
            mask_delta=mask_delta,  # opt.data_online_creation_mask_delta_A,
            crop_delta=0,
            mask_square=mask_square,  # opt.data_online_creation_mask_square_A,
            crop_dim=opt.data_online_creation_crop_size_A,  # we use the average crop_dim
            output_dim=opt.data_load_size,
            context_pixels=opt.data_online_context_pixels,
            load_size=opt.data_online_creation_load_size_A,
            crop_coordinates=crop_coordinates,
            crop_center=True,
        )

        x_crop, y_crop, crop_size = crop_coordinates

        bbox = bboxes[0]

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

    if img_width and img_height > 0:
        img = cv2.resize(img, (img_width, img_height))

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

    mask = torch.from_numpy(np.array(mask, dtype=np.int64)).unsqueeze(0)
    """if crop_width > 0 and crop_height > 0:
        mask = resize(mask).clone().detach()"""

    if not cpu:
        img_tensor = img_tensor.to(device).clone().detach()
        mask = mask.to(device).clone().detach()

    if opt.data_online_creation_rand_mask_A:
        y_t = fill_mask_with_random(
            img_tensor.clone().detach(), mask.clone().detach(), -1
        )
    elif opt.data_online_creation_color_mask_A:
        y_t = fill_mask_with_color(
            img_tensor.clone().detach(), mask.clone().detach(), {}
        )

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
    elif opt.alg_palette_cond_image_creation == "edges":
        cond_image = fill_img_with_edges(img_tensor.unsqueeze(0), mask.unsqueeze(0))

    # run through model
    if (
        reconstruction_guidance
        and previous_frame is not None
        and previous_frame_bbox_in is not None
    ):

        previous_frame_mask, _, _, _, _, _ = create_mask(
            previous_frame_bbox_in,
            select_cat=-1,
            shape=img_orig.shape,
            ratio_x=1,
            ratio_y=1,
            mask_delta=mask_delta,
            mask_square=mask_square,
            context_pixels=opt.data_online_context_pixels,
        )

        """crop_coordinates = crop_image(
            img_path=img_in,
            bbox_path=bbox_in,
            mask_delta=mask_delta,  # =opt.data_online_creation_mask_delta_A,
            crop_delta=0,
            mask_square=mask_square,  # opt.data_online_creation_mask_square_A,
            crop_dim=opt.data_online_creation_crop_size_A,  # we use the average crop_dim
            output_dim=opt.data_load_size,
            context_pixels=opt.data_online_context_pixels,
            load_size=opt.data_online_creation_load_size_A,
            get_crop_coordinates=True,
            crop_center=True,
        )"""

        previous_frame_mask = previous_frame_mask[
            bbox_select[1] : bbox_select[3], bbox_select[0] : bbox_select[2]
        ]

        previous_frame_mask = cv2.resize(
            previous_frame_mask,
            (opt.data_load_size, opt.data_load_size),
            interpolation=cv2.INTER_NEAREST,
        )

        previous_frame_mask = torch.tensor(previous_frame_mask)

        previous_frame_mask = torch.clamp(
            previous_frame_mask.to(device).clone().detach().unsqueeze(0), min=0, max=1
        )

        output_previous, _ = model.restoration(
            y_cond=-1 * torch.ones_like(cond_image, device=cond_image.device),
            y_t=previous_frame,
            y_0=previous_frame,  # only used for background so we can take the previous frame generated or grount truth
            mask=previous_frame_mask,
            sample_num=2,
        )
        x_a = output_previous
    else:
        x_a = None

    y_t, cond_image, img_tensor, mask = (
        y_t.unsqueeze(0).clone().detach(),
        cond_image.clone().detach(),
        img_tensor.unsqueeze(0).clone().detach(),
        torch.clamp(mask, min=0, max=1).unsqueeze(0).clone().detach(),
    )

    out_tensor, visu = model.restoration(
        y_cond=cond_image, y_t=y_t, y_0=img_tensor, mask=mask, sample_num=2, x_a=x_a
    )

    out_img = to_np(
        out_tensor
    )  # out_img = out_img.detach().data.cpu().float().numpy()[0]

    """ post-processing
    
    out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)"""

    if img_width > 0 or img_height > 0 or crop_width > 0 or crop_height > 0:
        # img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)

        out_img_resized = cv2.resize(
            out_img,
            (
                min(img_orig.shape[1], bbox_select[2] - bbox_select[0]),
                min(img_orig.shape[0], bbox_select[3] - bbox_select[1]),
            ),
        )

    out_img_real_size = img_orig.copy()

    # fill out crop into original image
    out_img_real_size[
        bbox_select[1] : bbox_select[3], bbox_select[0] : bbox_select[2]
    ] = out_img_resized

    cond_img = to_np(cond_image)

    if write:
        cv2.imwrite(os.path.join(dir_out, name + "_orig.png"), img_orig)
        cv2.imwrite(os.path.join(dir_out, name + "_generated_crop.png"), out_img)
        cv2.imwrite(os.path.join(dir_out, name + "_cond.png"), cond_img)
        cv2.imwrite(os.path.join(dir_out, name + "_generated.png"), out_img_real_size)
        cv2.imwrite(os.path.join(dir_out, name + "_y_0.png"), to_np(img_tensor))
        cv2.imwrite(os.path.join(dir_out, name + "_y_t.png"), to_np(y_t))
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
        "--previous-frame-bbox-in", help="image to transform", default=None
    )

    options.parser.add_argument(
        "--mask-in", help="mask used for image transformation", required=False
    )
    options.parser.add_argument("--bbox-in", help="bbox file used for masking")

    options.parser.add_argument(
        "--nb_samples", help="nb of samples generated", type=int, default=1
    )

    args = options.parse()

    args.write = True

    real_name = args.name

    for i in tqdm(range(args.nb_samples)):
        args.name = real_name + "_" + str(i).zfill(len(str(args.nb_samples)))
        generate(**vars(args))
