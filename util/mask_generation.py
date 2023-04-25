import os
import random
import sys
import urllib.request

import cv2
import numpy as np
import requests
import torch
from torch.nn import functional as F
from torchvision.transforms import Grayscale

from models.modules.sam.sam_inference import predict_sam_edges
from models.modules.sketch_generation.hed import HEDdetector
from models.modules.sketch_generation.mlsd import MLSDdetector
from models.modules.utils import download_midas_weight, predict_depth
from util.util import im2tensor

sys.path.append("./../")


def fill_img_with_sketch(img, mask, **kwargs):
    """Fill the masked region with sketch edges."""

    grayscale = Grayscale(3)
    gray = grayscale(img)

    threshold = torch.tensor((120 / 255) * 2 - 1)

    thresh = (gray < threshold) * 1.0  # thresh = ((gray < threshold) * 1.0) * 2 - 1

    mask = torch.clamp(mask, 0, 1)

    return mask * thresh + (1 - mask) * img


def fill_img_with_canny(
    img,
    mask,
    low_threshold=None,
    high_threshold=None,
    **kwargs,
):
    """Fill the masked region with canny edges."""
    low_threshold_random = kwargs["low_threshold_random"]
    high_threshold_random = kwargs["high_threshold_random"]
    max_value = 255 * 3
    if high_threshold is None and low_threshold is None:
        threshold_1 = random.randint(low_threshold_random, high_threshold_random)
        threshold_2 = random.randint(low_threshold_random, high_threshold_random)
        high_threshold = max(threshold_1, threshold_2)
        low_threshold = min(threshold_1, threshold_2)
    elif high_threshold is None and low_threshold is not None:
        high_threshold = random.randint(low_threshold, max_value)
    elif high_threshold is not None and low_threshold is None:
        low_threshold = random.randint(0, high_threshold)

    device = img.device
    edges_list = []
    for cur_img in img:
        cur_img = (
            (torch.einsum("chw->hwc", cur_img).cpu().numpy() + 1) * 255 / 2
        ).astype(np.uint8)
        edges = cv2.Canny(cur_img, low_threshold, high_threshold)
        edges = (
            (((torch.tensor(edges, device=device) / 255) * 2) - 1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        edges_list.append(edges)
    edges = torch.cat(edges_list, dim=0)
    mask = torch.clamp(mask, 0, 1)

    return mask * edges + (1 - mask) * img


def fill_img_with_hed(img, mask, **kwargs):
    """Fill the masked region with HED edges from the ControlNet paper."""

    apply_hed = HEDdetector()
    device = img.device
    edges_list = []
    for cur_img in img:
        cur_img = (
            (torch.einsum("chw->hwc", cur_img).cpu().numpy() + 1) * 255 / 2
        ).astype(np.uint8)
        detected_map = apply_hed(cur_img)
        detected_map = (
            (((torch.tensor(detected_map, device=device) / 255) * 2) - 1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        edges_list.append(detected_map)
    edges = torch.cat(edges_list, dim=0)
    mask = torch.clamp(mask, 0, 1)

    return mask * edges + (1 - mask) * img


def fill_img_with_hough(
    img,
    mask,
    value_threshold=1e-05,
    distance_threshold=10.0,
    with_canny=False,
    **kwargs,
):
    """Fill the masked region with Hough lines detection from the ControlNet paper."""

    if with_canny:
        img = fill_img_with_canny(img, mask, **kwargs)

    device = img.device
    apply_mlsd = MLSDdetector()
    edges_list = []
    for cur_img in img:
        cur_img = (
            (torch.einsum("chw->hwc", cur_img).cpu().numpy() + 1) * 255 / 2
        ).astype(np.uint8)
        detected_map = apply_mlsd(
            cur_img, thr_v=value_threshold, thr_d=distance_threshold
        )
        detected_map = (
            (((torch.tensor(detected_map, device=device) / 255) * 2) - 1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        edges_list.append(detected_map)
    edges = torch.cat(edges_list, dim=0)
    mask = torch.clamp(mask, 0, 1)

    return mask * edges + (1 - mask) * img


def fill_img_with_depth(img, mask, depth_network="DPT_SwinV2_T_256", **kwargs):
    """Fill the masked region with depth map."""

    device = img.device
    midas_w = download_midas_weight(model_type=depth_network)
    edges_list = []
    for cur_img in img:
        cur_img = torch.from_numpy(
            np.transpose(
                ((torch.einsum("chw->hwc", cur_img).cpu() + 1) * 255 / 2).numpy(),
                (2, 0, 1),
            )
        ).float()
        depth_map = predict_depth(
            img=cur_img.unsqueeze(0), midas=midas_w, model_type=depth_network
        )
        if (depth_map.shape[0], depth_map.shape[1]) != (
            cur_img.shape[1],
            cur_img.shape[2],
        ):
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=cur_img.shape[1:],
                mode="bilinear",
            ).squeeze()
        depth_map = (
            ((torch.tensor(depth_map, device=device) / 255) * 2) - 1
        ).unsqueeze(0)
        edges_list.append(depth_map)
    edges = torch.cat(edges_list, dim=0)
    mask = torch.clamp(mask, 0, 1)

    return mask * edges + (1 - mask) * img


def fill_img_with_sam(img, mask, sam, opt):
    crops = []
    for i, cur_img in enumerate(img):
        cur_img = (
            (torch.einsum("chw->hwc", cur_img).cpu().numpy() + 1) * 255 / 2
        ).astype(np.uint8)
        cur_mask = mask[i].cpu().squeeze(0).numpy()
        cur_mask[cur_mask > 0] = 1
        bbox = np.argwhere(cur_mask == 1)
        crop_delta = opt.alg_palette_sam_crop_delta
        if bbox.shape[0] != 0:
            x_min = np.min(bbox[:, 0])
            x_max = np.max(bbox[:, 0])
            y_min = np.min(bbox[:, 1])
            y_max = np.max(bbox[:, 1])
        else:
            x_min = 0
            x_max = cur_img.shape[0]
            y_min = 0
            y_max = cur_img.shape[1]
        x_min_to_crop = max(0, x_min - crop_delta)
        x_max_to_crop = min(cur_img.shape[0], x_max + crop_delta)
        y_min_to_crop = max(0, y_min - crop_delta)
        y_max_to_crop = min(cur_img.shape[1], y_max + crop_delta)
        img_cropped = cur_img[
            x_min_to_crop:x_max_to_crop, y_min_to_crop:y_max_to_crop, :
        ]
        crops.append(
            (
                img_cropped,
                (x_min, x_max, y_min, y_max),
                (x_min_to_crop, x_max_to_crop, y_min_to_crop, y_max_to_crop),
            )
        )
    edges_list = predict_sam_edges(
        [crop[0] for crop in crops],
        sam,
        use_gaussian_filter=opt.alg_palette_sam_use_gaussian_filter,
        use_sobel_filter=opt.alg_palette_sam_no_sobel_filter,
        output_binary_sam=opt.alg_palette_sam_no_output_binary_sam,
        redundancy_threshold=opt.alg_palette_sam_redundancy_threshold,
        sobel_threshold=opt.alg_palette_sam_sobel_threshold,
        final_canny=opt.alg_palette_sam_final_canny,
        min_mask_area=opt.alg_palette_sam_min_mask_area,
        max_mask_area=opt.alg_palette_sam_max_mask_area,
        points_per_side=opt.alg_palette_sam_points_per_side,
        sample_points_in_ellipse=opt.alg_palette_sam_no_sample_points_in_ellipse,
    )

    output_list = []
    for k in range(len(img)):
        edges = edges_list[k]
        x_min, x_max, y_min, y_max = crops[k][1]
        x_min_to_crop, x_max_to_crop, y_min_to_crop, y_max_to_crop = crops[k][2]

        crop_x_min = abs(x_min_to_crop - x_min)
        crop_x_max = abs(x_max_to_crop - x_max)
        crop_y_min = abs(y_min_to_crop - y_min)
        crop_y_max = abs(y_max_to_crop - y_max)

        cur_img = (
            (torch.einsum("chw->hwc", img[k]).cpu().numpy() + 1) * 255 / 2
        ).astype(np.uint8)

        cur_img[x_min:x_max, y_min:y_max, :] = edges[
            crop_x_min : edges.shape[0] - crop_x_max,
            crop_y_min : edges.shape[1] - crop_y_max,
        ][..., np.newaxis]

        output_list.append(im2tensor(cur_img).unsqueeze(0).to(sam.device))

    output_list = torch.cat(output_list, dim=0)

    return output_list


def random_edge_mask(fn_list):
    edge_fns = []
    for fn in fn_list:
        if fn == "canny":
            edge_fns.append(fill_img_with_canny)
        elif fn == "hed":
            edge_fns.append(fill_img_with_hed)
        elif fn == "hough":
            edge_fns.append(fill_img_with_hough)
        elif fn == "depth":
            edge_fns.append(fill_img_with_depth)
        elif fn == "sketch":
            edge_fns.append(fill_img_with_sketch)
        elif fn == "sam":
            edge_fns.append(fill_img_with_sam)
        else:
            raise NotImplementedError(f"Unknown edge function {fn}")
    return random.choice(edge_fns)
