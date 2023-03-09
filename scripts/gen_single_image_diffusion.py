import argparse
import json
import math
import os
import random
import re
import sys
import warnings

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Resize
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append("../")


from diffusion_options import DiffusionOptions

from data.online_creation import crop_image, fill_mask_with_color, fill_mask_with_random
from models import diffusion_networks
from models.modules.diffusion_utils import set_new_noise_schedule
from options.train_options import TrainOptions
from util.mask_generation import (
    fill_img_with_canny,
    fill_img_with_depth,
    fill_img_with_edges,
    fill_img_with_hed,
    fill_img_with_hed_Caffe,
    fill_img_with_hough,
    fill_img_with_sketch,
)


def load_model(modelpath, model_in_file, device, sampling_steps, is_norm):
    train_json_path = modelpath + "train_config.json"
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
    expected_keys = model.state_dict()
    if is_norm:
        sd = torch.load(modelpath + model_in_file)
        for key in list(sd.keys()):
            if (
                key.startswith("denoise_fn.input_blocks")
                or key.startswith("denoise_fn.output_blocks")
                or key.startswith("denoise_fn.middle_block")
                or key.startswith("denoise_fn.out")
            ):
                if key.endswith(".weight") or key.endswith(".bias"):
                    match = re.search(r"\d+.(norm|bias).", key)
                    if not match:
                        # Modify the key to insert ".norm" just before the
                        # ".weight" or ".bias"
                        new_key = key.replace(".weight", ".norm.weight").replace(
                            ".bias", ".norm.bias"
                        )
                        # Move the value to the new key and delete the old key
                        sd[new_key] = sd[key]
    else:
        sd = torch.load(modelpath + model_in_file)

    state_dict = {k: v for k, v in sd.items() if k in expected_keys}

    model.load_state_dict(state_dict)

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


def resize_bbox(
    bbox, mask_delta, mask_square, opt, x_crop=None, y_crop=None, crop_size=None
):
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

    if x_crop is not None and y_crop is not None and crop_size is not None:
        bbox_select[1] += y_crop
        bbox_select[0] += x_crop

        bbox_select[3] = bbox_select[1] + crop_size
        bbox_select[2] = bbox_select[0] + crop_size

    bbox_select[1] -= opt.data_online_context_pixels
    bbox_select[0] -= opt.data_online_context_pixels

    bbox_select[3] += opt.data_online_context_pixels
    bbox_select[2] += opt.data_online_context_pixels

    return bbox_select


def transfer_cond(
    img_tensor,
    bbox,
    source_img,
    source_bbox,
    source_id,
    x_crop,
    y_crop,
    mask_delta,
    mask_square,
    opt,
    device,
):
    if source_bbox is not None:
        # img_bus_path = '/data3/beniz/data/mapillary/images/04dkjiUmbNagx0nBB8VZJQ.jpg'
        img_bus_path = source_img  # '/data3/beniz/data/mapillary/images/DSdC0LB71avSiBqw1r9kJg.jpg'
        bbox_bus_path = source_bbox  # '/data3/beniz/data/mapillary/dd_cls/partially/bbox/DSdC0LB71avSiBqw1r9kJg.txt'
        # bbox_bus_path = '/data3/beniz/data/mapillary/dd_cls/partially/bbox/04dkjiUmbNagx0nBB8VZJQ.txt'
        crop_coordinates_bus = crop_image(
            img_path=img_bus_path,
            bbox_path=bbox_bus_path,
            mask_delta=mask_delta,  # =opt.data_online_creation_mask_delta_A,
            crop_delta=0,
            mask_square=mask_square,  # opt.data_online_creation_mask_square_A,
            mask_random_offset=None,
            crop_dim=opt.data_online_creation_crop_size_A,  # we use the average crop_dim
            output_dim=opt.data_load_size,
            context_pixels=opt.data_online_context_pixels,
            load_size=opt.data_online_creation_load_size_A,
            get_crop_coordinates=True,
            crop_center=True,
            bbox_id=source_id,
        )
        img_bus, mask_bus = crop_image(
            img_path=img_bus_path,
            bbox_path=bbox_bus_path,
            mask_delta=mask_delta,  # opt.data_online_creation_mask_delta_A,
            crop_delta=0,
            mask_square=mask_square,  # opt.data_online_creation_mask_square_A,
            mask_random_offset=None,
            crop_dim=opt.data_online_creation_crop_size_A,  # we use the average crop_dim
            output_dim=opt.data_load_size,
            context_pixels=opt.data_online_context_pixels,
            load_size=opt.data_online_creation_load_size_A,
            crop_coordinates=crop_coordinates_bus,
            crop_center=True,
            bbox_id=source_id,
        )
        x_crop_bus, y_crop_bus, crop_size_bus = crop_coordinates_bus
        with open(bbox_bus_path, "r") as f:
            bbox_bus = f.readlines()[0].split()[1:]
        bbox_bus = [int(i) for i in bbox_bus]
        bbox_select_bus = resize_bbox(bbox_bus, mask_delta, mask_square, opt)

        img_bus, mask_bus = np.array(img_bus), np.array(mask_bus)

        img_tensor_bus = tran(img_bus).clone().detach()
        mask_bus = torch.from_numpy(np.array(mask_bus, dtype=np.int64)).unsqueeze(0)
        if not cpu:
            img_tensor_bus = img_tensor_bus.to(device).clone().detach()
            mask_bus = mask_bus.to(device).clone().detach()

        if opt.alg_palette_cond_image_creation == "sketch":
            cond_image_bus = fill_img_with_sketch(
                img_tensor_bus.unsqueeze(0), mask_bus.unsqueeze(0)
            )
        elif opt.alg_palette_cond_image_creation == "canny":
            cond_image_bus = fill_img_with_canny(
                img_tensor_bus.unsqueeze(0), mask_bus.unsqueeze(0)
            )
        elif opt.alg_palette_cond_image_creation == "edges":
            cond_image_bus = fill_img_with_edges(
                img_tensor_bus.unsqueeze(0), mask_bus.unsqueeze(0)
            )
        elif opt.alg_palette_cond_image_creation == "hed":
            cond_image_bus = fill_img_with_hed(
                img_tensor_bus.unsqueeze(0), mask_bus.unsqueeze(0)
            )
        elif opt.alg_palette_cond_image_creation == "hough":
            cond_image_bus = fill_img_with_hough(
                img_tensor_bus.unsqueeze(0), mask_bus.unsqueeze(0)
            )
        elif opt.alg_palette_cond_image_creation == "depth":
            cond_image_bus = fill_img_with_depth(
                img_tensor_bus.unsqueeze(0), mask_bus.unsqueeze(0)
            )

        cond_image = img_tensor.clone().detach()
        ##- resize to [1, 3, w, h]
        cond_image = cond_image[None, :, :, :]
        bbox_orig = resize_bbox(bbox, mask_delta, mask_square, opt)

        bbox_orig_w = bbox_orig[2] - bbox_orig[0]
        bbox_orig_h = bbox_orig[3] - bbox_orig[1]
        bbox_w = bbox_select_bus[2] - bbox_select_bus[0]
        bbox_h = bbox_select_bus[3] - bbox_select_bus[1]

        if mask_square:
            bbox_orig_w = max(bbox_orig_h, bbox_orig_w)
            bbox_orig_h = max(bbox_orig_h, bbox_orig_w)
            bbox_w = max(bbox_h, bbox_w)
            bbox_h = max(bbox_h, bbox_w)
            left_side = int((crop_size_bus - bbox_w) / 2)
            right_side = crop_size_bus - left_side
            up_side = int((crop_size_bus - bbox_h) / 2)

            down_side = crop_size_bus - up_side
        else:
            left_side = abs(x_crop_bus)
            right_side = left_side + bbox_w
            up_side = abs(y_crop_bus)
            down_side = up_side + bbox_h

        mask_2D_bus = mask_bus.cpu()[0, :, :]  # Convert mask to a 2D array
        coords = np.column_stack(
            np.where(mask_2D_bus > 0)
        )  # Get the coordinates of the white pixels in the mask

        x_0, y_0, w, h = cv2.boundingRect(coords)

        left_side_orig = abs(x_crop)
        right_side_orig = left_side_orig + bbox_orig_w
        up_side_orig = abs(y_crop)
        down_side_orig = up_side_orig + bbox_orig_h

        bus_sketch = cond_image_bus[:, :, y_0 : y_0 + h, x_0 : x_0 + w]

        resizer = Resize((bbox_orig_h, bbox_orig_w))
        resized_bus_sketch = resizer(bus_sketch)

        cond_image[
            :, :, up_side_orig:down_side_orig, left_side_orig:right_side_orig
        ] = resized_bus_sketch
        return cond_image
    else:
        bbox_orig = resize_bbox(bbox, mask_delta, mask_square, opt)
        bbox_orig_w = bbox_orig[2] - bbox_orig[0]
        bbox_orig_h = bbox_orig[3] - bbox_orig[1]

        if bbox_orig_w >= 256:
            bbox_orig_w = 256
            x_crop = 0
        if bbox_orig_h >= 256:
            y_crop = 0
            bbox_orig_h = 256

        left_side_orig = abs(x_crop)
        right_side_orig = left_side_orig + bbox_orig_w
        up_side_orig = abs(y_crop)
        down_side_orig = up_side_orig + bbox_orig_h

        cond_image = img_tensor.clone().detach()
        cond_image = cond_image[None, :, :, :]

        transfer_img = cv2.imread(source_img)
        transfer_tensor = torch.from_numpy(transfer_img).unsqueeze(0)
        transfer_tensor = transfer_tensor.permute(0, 3, 1, 2)
        transfer_tensor = transfer_tensor / 255.0
        white_mask = torch.all(transfer_tensor >= 0.9, dim=1)

        # Set non-white pixels to 0
        transfer_tensor[:, 0][~white_mask] = 0.0
        transfer_tensor[:, 1][~white_mask] = 0.0
        transfer_tensor[:, 2][~white_mask] = 0.0

        resizer = Resize((bbox_orig_h, bbox_orig_w))
        resized_transfer_img = resizer(transfer_tensor)

        cond_image[
            :, :, up_side_orig:down_side_orig, left_side_orig:right_side_orig
        ] = resized_transfer_img

        return cond_image


def generate(
    seed,
    model_in_file,
    model_norm,
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
    name,
    mask_delta,
    mask_square,
    bbox_id,
    transfer,
    source_img,
    source_bbox,
    source_id,
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
        modelpath, os.path.basename(model_in_file), device, sampling_steps, model_norm
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
            bbox_orig = bboxes[bbox_id]
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
            mask_random_offset=None,
            crop_dim=opt.data_online_creation_crop_size_A,  # we use the average crop_dim
            output_dim=opt.data_load_size,
            context_pixels=opt.data_online_context_pixels,
            load_size=opt.data_online_creation_load_size_A,
            get_crop_coordinates=True,
            crop_center=True,
            bbox_id=bbox_id,
        )

        img, mask = crop_image(
            img_path=img_in,
            bbox_path=bbox_in,
            mask_delta=mask_delta,  # opt.data_online_creation_mask_delta_A,
            crop_delta=0,
            mask_square=mask_square,  # opt.data_online_creation_mask_square_A,
            mask_random_offset=None,
            crop_dim=opt.data_online_creation_crop_size_A,  # we use the average crop_dim
            output_dim=opt.data_load_size,
            context_pixels=opt.data_online_context_pixels,
            load_size=opt.data_online_creation_load_size_A,
            crop_coordinates=crop_coordinates,
            crop_center=True,
            bbox_id=bbox_id,
        )

        x_crop, y_crop, crop_size = crop_coordinates

        bbox = bboxes[bbox_id]

        bbox_select = resize_bbox(
            bbox, mask_delta, mask_square, opt, x_crop, y_crop, crop_size
        )

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
        if transfer:
            cond_image = transfer_cond(
                img_tensor,
                bbox,
                source_img,
                source_bbox,
                source_id,
                x_crop,
                y_crop,
                mask_delta,
                mask_square,
                opt,
                device,
            )
        else:
            cond_image = fill_img_with_sketch(
                img_tensor.unsqueeze(0), mask.unsqueeze(0)
            )
    elif opt.alg_palette_cond_image_creation == "canny":
        if transfer:
            cond_image = transfer_cond(
                img_tensor,
                bbox,
                source_img,
                source_bbox,
                source_id,
                x_crop,
                y_crop,
                mask_delta,
                mask_square,
                opt,
                device,
            )
        else:
            cond_image = fill_img_with_canny(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_palette_cond_image_creation == "hed":
        if transfer:
            cond_image = transfer_cond(
                img_tensor,
                bbox,
                source_img,
                source_bbox,
                source_id,
                x_crop,
                y_crop,
                mask_delta,
                mask_square,
                opt,
                device,
            )
        else:
            cond_image = fill_img_with_hed(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_palette_cond_image_creation == "hough":
        if transfer:
            cond_image = transfer_cond(
                img_tensor,
                bbox,
                source_img,
                source_bbox,
                source_id,
                x_crop,
                y_crop,
                mask_delta,
                mask_square,
                opt,
                device,
            )
        else:
            cond_image = fill_img_with_hough(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_palette_cond_image_creation == "depth":
        if transfer:
            cond_image = transfer_cond(
                img_tensor,
                bbox,
                source_img,
                source_bbox,
                source_id,
                x_crop,
                y_crop,
                mask_delta,
                mask_square,
                opt,
                device,
            )
        else:
            cond_image = fill_img_with_depth(img_tensor.unsqueeze(0), mask.unsqueeze(0))
    elif opt.alg_palette_cond_image_creation == "edges":
        if transfer:
            cond_image = transfer_cond(
                img_tensor,
                bbox,
                source_img,
                source_bbox,
                source_id,
                x_crop,
                y_crop,
                mask_delta,
                mask_square,
                opt,
                device,
            )
        else:
            cond_image = fill_img_with_edges(img_tensor.unsqueeze(0), mask.unsqueeze(0))

    # run through model
    y_t, cond_image, img_tensor, mask = (
        y_t.unsqueeze(0).clone().detach(),
        cond_image.clone().detach(),
        img_tensor.unsqueeze(0).clone().detach(),
        torch.clamp(mask, min=0, max=1).unsqueeze(0).clone().detach(),
    )

    with torch.no_grad():
        out_tensor, visu = model.restoration(
            y_cond=cond_image, y_t=y_t, y_0=img_tensor, mask=mask, sample_num=2
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

    bbox_select[0] = max(0, bbox_select[0])
    bbox_select[2] = max(crop_size, bbox_select[2])
    bbox_select[1] = max(0, bbox_select[1])
    bbox_select[3] = max(crop_size, bbox_select[3])
    # fill out crop into original image
    out_img_real_size[
        bbox_select[1] : bbox_select[3], bbox_select[0] : bbox_select[2]
    ] = out_img_resized

    cond_img = to_np(cond_image)
    if write:
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        cv2.imwrite(os.path.join(dir_out, name + "_orig.jpg"), img_orig)
        cv2.imwrite(os.path.join(dir_out, name + "_generated_crop.jpg"), out_img)
        cv2.imwrite(os.path.join(dir_out, name + "_cond.jpg"), cond_img)
        cv2.imwrite(os.path.join(dir_out, name + "_generated.jpg"), out_img_real_size)
        cv2.imwrite(os.path.join(dir_out, name + "_y_0.jpg"), to_np(img_tensor))
        cv2.imwrite(os.path.join(dir_out, name + "_y_t.jpg"), to_np(y_t))
        cv2.imwrite(os.path.join(dir_out, name + "_mask.jpg"), to_np(mask))

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

    args = options.parse()

    args.write = True

    real_name = args.name

    for i in tqdm(range(args.nb_samples)):
        args.name = real_name + "_" + str(i).zfill(len(str(args.nb_samples)))
        generate(**vars(args))
