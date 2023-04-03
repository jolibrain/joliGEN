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

from models.modules.sketch_generation.hed import HEDdetector
from models.modules.sketch_generation.mlsd import MLSDdetector
from models.modules.utils import download_midas_weight, predict_depth

sys.path.append("./../")


def fill_img_with_sketch(img, mask):
    """Fill the masked region with sketch edges."""

    grayscale = Grayscale(3)
    gray = grayscale(img)

    threshold = torch.tensor((120 / 255) * 2 - 1)

    thresh = (gray < threshold) * 1.0  # thresh = ((gray < threshold) * 1.0) * 2 - 1

    mask = torch.clamp(mask, 0, 1)

    return mask * thresh + (1 - mask) * img


def fill_img_with_canny(img, mask, low_threshold=150, high_threshold=200):
    """Fill the masked region with canny edges."""

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


def fill_img_with_hed(img, mask):
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
    img, mask, value_threshold=1e-05, distance_threshold=10.0, with_canny=False
):
    """Fill the masked region with Hough lines detection from the ControlNet paper."""

    if with_canny:
        img = fill_img_with_canny(img, mask)

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


def fill_img_with_depth(img, mask, depth_network="DPT_SwinV2_T_256"):
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
        depth_map = (
            ((torch.tensor(depth_map, device=device) / 255) * 2) - 1
        ).unsqueeze(0)
        edges_list.append(depth_map)
    edges = torch.cat(edges_list, dim=0)
    mask = torch.clamp(mask, 0, 1)

    return mask * edges + (1 - mask) * img


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
        else:
            raise NotImplementedError(f"Unknown edge function {fn}")
    return random.choice(edge_fns)
