import sys
import os
import json
import random

sys.path.append("../")
from models import diffusion_networks
from options.train_options import TrainOptions
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import argparse
from data.online_creation import fill_mask_with_random, fill_mask_with_color
from models.modules.diffusion_utils import set_new_noise_schedule
from util.mask_generation import fill_img_with_sketch, fill_img_with_edges
import warnings


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
    model.eval()
    model.load_state_dict(torch.load(modelpath + "/" + model_in_file))

    # sampling steps
    if sampling_steps > 0:
        model.denoise_fn.beta_schedule["test"]["n_timestep"] = sampling_steps
        set_new_noise_schedule(model.denoise_fn, "test")

    model = model.to(device)
    return model, opt


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
    img_out,
    **unused_options,
):

    # seed
    if seed >= 0:
        torch.manual_seed(seed)

    # loading model
    modelpath = model_in_file.replace(os.path.basename(model_in_file), "")
    print("modelpath=", modelpath)

    if not cpu:
        device = torch.device("cuda:" + str(gpuid))
    else:
        device = torch.device("cpu")
    model, opt = load_model(
        modelpath, os.path.basename(model_in_file), device, sampling_steps
    )

    # reading image
    img = cv2.imread(img_in)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # reading the mask
    if mask_in:
        mask = cv2.imread(mask_in, 0)

    bboxes = []
    if bbox_in:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        with open(bbox_in, "r") as bboxf:
            for line in bboxf:
                elts = line.rstrip().split()
                bboxes.append([int(elts[1]), int(elts[2]), int(elts[3]), int(elts[4])])

        if crop_width or crop_height > 0:
            hc_width = int(crop_width / 2)
            hc_height = int(crop_height / 2)
            # select one bbox and crop around it
            bbox_orig = random.choice(bboxes)
            if args.bbox_width_factor > 0.0:
                bbox_orig[0] -= max(0, int(args.bbox_width_factor * bbox_orig[0]))
                bbox_orig[2] += max(0, int(args.bbox_width_factor * bbox_orig[2]))
            if args.bbox_height_factor > 0.0:
                bbox_orig[1] -= max(0, int(args.bbox_height_factor * bbox_orig[1]))
                bbox_orig[3] += max(0, int(args.bbox_height_factor * bbox_orig[3]))

            bbox_select = bbox_orig.copy()
            bbox_select[0] -= max(0, hc_width)
            bbox_select[0] = max(0, bbox_select[0])
            bbox_select[1] -= max(0, hc_height)
            bbox_select[1] = max(0, bbox_select[1])
            bbox_select[2] += hc_width
            bbox_select[2] = min(img.shape[1], bbox_select[2])
            bbox_select[3] += hc_height
            bbox_select[3] = min(img.shape[0], bbox_select[3])

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
            ]  # cropped img
        else:
            for bbox in bboxes:
                mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = np.full(
                    (bbox[3] - bbox[1], bbox[2] - bbox[0]), 1
                )  # ymin:ymax, xmin:xmax, ymax-ymin, xmax-xmin

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
    if crop_width > 0 and crop_height > 0:
        mask = resize(mask).clone().detach()

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

    if opt.alg_palette_cond_image_creation == "y_t":
        cond_image = y_t
    elif opt.alg_palette_cond_image_creation == "sketch":
        cond_image = fill_img_with_sketch(img_tensor, mask)
    elif opt.alg_palette_cond_image_creation == "edges":
        cond_image = fill_img_with_edges(img_tensor, mask)

    # run through model
    y_t, cond_image, img_tensor, mask = (
        y_t.unsqueeze(0).clone().detach(),
        cond_image.unsqueeze(0).clone().detach(),
        img_tensor.unsqueeze(0).clone().detach(),
        mask.unsqueeze(0).clone().detach(),
    )

    with torch.no_grad():

        out_tensor, visu = model.restoration(
            y_cond=cond_image, y_t=y_t, y_0=img_tensor, mask=mask, sample_num=2
        )

    # post-processing
    out_img = out_tensor.detach().data.cpu().float().numpy()[0]
    out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    if crop_width > 0 or crop_height > 0:
        # fill out crop into original image
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
        out_img = cv2.resize(
            out_img,
            (
                min(img_orig.shape[1], bbox_select[2] - bbox_select[0]),
                min(img_orig.shape[0], bbox_select[3] - bbox_select[1]),
            ),
        )
        cv2.imwrite(img_out + "_crop.png", out_img)
        img_orig[
            bbox_select[1] : bbox_select[3], bbox_select[0] : bbox_select[2]
        ] = out_img
        out_img = img_orig.copy()

    if opt.alg_palette_cond_image_creation == "sketch":
        cond_img = cond_image.detach().data.cpu().float().numpy()[0]
        cond_img = (np.transpose(cond_img, (1, 2, 0)) + 1) / 2.0 * 255.0
        cond_img = cv2.cvtColor(cond_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_out + "_cond_mask.png", cond_img)

    cv2.imwrite(img_out, out_img)

    print("Successfully generated image ", img_out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-in-file",
        help="file path to generator model (.pth file)",
        required=True,
    )

    parser.add_argument("--img-width", default=-1, type=int, help="image width")
    parser.add_argument("--img-height", default=-1, type=int, help="image height")

    parser.add_argument(
        "--crop-width",
        default=-1,
        type=int,
        help="crop width added on each side of the bbox (optional)",
    )
    parser.add_argument(
        "--crop-height",
        default=-1,
        type=int,
        help="crop height added on each side of the bbox (optional)",
    )

    parser.add_argument("--img-in", help="image to transform", required=True)
    parser.add_argument(
        "--mask-in", help="mask used for image transformation", required=False
    )
    parser.add_argument("--bbox-in", help="bbox file used for masking")
    parser.add_argument(
        "--bbox-width-factor",
        type=float,
        default=0.0,
        help="bbox width added factor of original width",
    )
    parser.add_argument(
        "--bbox-height-factor",
        type=float,
        default=0.0,
        help="bbox height added factor of original height",
    )
    parser.add_argument("--img-out", help="transformed image", required=True)
    parser.add_argument(
        "--sampling-steps", default=-1, type=int, help="number of sampling steps"
    )
    parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
    parser.add_argument("--gpuid", type=int, default=0, help="which GPU to use")
    parser.add_argument(
        "--seed", type=int, default=-1, help="random seed for reproducibility"
    )
    args = parser.parse_args()

    generate(**vars(args))
