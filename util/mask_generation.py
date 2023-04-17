import os
import random
import sys
import urllib.request

import cv2
import numpy as np
import requests
import scipy
import torch
from torch.nn import functional as F
from torchvision.transforms import Grayscale

from models.modules.sketch_generation.hed import HEDdetector
from models.modules.sketch_generation.mlsd import MLSDdetector
from models.modules.utils import download_midas_weight, predict_depth

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


def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def non_max_suppression(masks, scores, threshold):
    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = []
    for i in sorted_indices:
        mask_i = masks[i]
        overlap = False
        for j in selected_indices:
            mask_j = masks[j]
            if iou(mask_i, mask_j) > threshold:
                overlap = True
                break
        if not overlap:
            selected_indices.append(i)
    selected_masks = [masks[i] for i in selected_indices]
    return selected_masks


def sam_edge_detection(image, predictor, redundancy_threshold):
    predictor.set_image(image)
    # create a 16x16 regular grid of foreground points as explained in the paper
    points = []
    for i in range(16):
        for j in range(16):
            points.append([i * image.shape[0] // 16, j * image.shape[1] // 16])
    masks = []
    scores = []
    logits = None
    for point in points:
        x, y = point
        masks_out, scores_out, logits_out = predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        for mask in masks_out:
            masks.append(mask)
        for score in scores_out:
            scores.append(score)
        if logits is None:
            logits = logits_out
        else:
            logits = np.maximum(logits, logits_out)

    masks = np.array(masks)
    scores = np.array(scores)

    non_redundant_masks = non_max_suppression(masks, scores, redundancy_threshold)
    non_redundant_masks = np.array(non_redundant_masks)

    masked_imgs = []
    for mask in non_redundant_masks:
        assert mask.shape == image.shape[:2], "mask should be the same size of image"
        prob_map = mask.astype(np.float32)
        prob_map /= 255.0
        # Apply Gaussian filter to probability map
        sigma = 2.0  # adjust sigma to control amount of smoothing
        prob_map = scipy.ndimage.gaussian_filter(prob_map, sigma=sigma)
        # Apply Sobel filter to probability map
        sobel_x = cv2.Sobel(prob_map, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(prob_map, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        # Apply non-maximum suppression to gradient magnitude
        threshold = 0.7 * np.max(
            grad_mag
        )  # set threshold at 70% of max gradient magnitude
        edge_map = (grad_mag > threshold).astype(np.uint8)
        # Find contours of mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            # Find outermost contour using convex hull
            hull = cv2.convexHull(contours[0])

            # Create binary mask of outer boundary pixels
            boundary_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(boundary_mask, [hull], -1, 255, -1)

            # Set values outside boundary to zero
            edge_map[~np.logical_and(edge_map, boundary_mask)] = 0

            # Threshold edge map to create binary mask
            threshold = 0.1  # adjust threshold to control amount of edge detection
            _, binary_map = cv2.threshold(edge_map, threshold, 255, cv2.THRESH_BINARY)
            binary_map = binary_map.astype(np.uint8)

            # Apply binary mask to original input image
            masked_img = cv2.bitwise_and(image, image, mask=binary_map)
            if len(masked_img[masked_img >= 1]) > 0:
                masked_imgs.append(masked_img)

    masked_imgs = np.array(masked_imgs)
    # Take pixel-wise max over all masked images
    final_pred = np.max(masked_imgs, axis=0)
    # Linearly normalize final prediction to range [0, 1]
    normalized_pred = (final_pred - np.min(final_pred)) / (
        np.max(final_pred) - np.min(final_pred)
    )
    # Apply edge nms (=Canny) to thicken edges
    threshold1 = 150
    threshold2 = 300
    edges = cv2.Canny(np.uint8(normalized_pred * 255), threshold1, threshold2)

    return edges


def fill_img_with_sam(img, mask, **predictor):
    predictor = predictor["predictor"]
    edges_list = []
    for cur_img in img:
        cur_img = (
            (torch.einsum("chw->hwc", cur_img).cpu().numpy() + 1) * 255 / 2
        ).astype(np.uint8)

        edges = sam_edge_detection(cur_img, predictor, redundancy_threshold=0.8)
        edges = (
            (((torch.tensor(edges, device=predictor.model.device) / 255) * 2) - 1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        edges_list.append(edges)
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
        elif fn == "sam":
            edge_fns.append(fill_img_with_sam)
        else:
            raise NotImplementedError(f"Unknown edge function {fn}")
    return random.choice(edge_fns)
