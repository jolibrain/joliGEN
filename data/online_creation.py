import math
import os
import random
import warnings

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from data.utils import load_image


def _scale_pixel_mask_delta(mask_delta, scale):
    if mask_delta == [[]]:
        return mask_delta

    scaled = []
    for delta in mask_delta:
        scaled_delta = []
        for value in delta:
            if isinstance(value, int):
                scaled_delta.append(int(round(value * scale)))
            else:
                scaled_delta.append(value)
        scaled.append(scaled_delta)
    return scaled


def _fit_rect_to_image(xmin, ymin, xmax, ymax, img_width, img_height):
    width = max(1, int(math.ceil(xmax - xmin)))
    height = max(1, int(math.ceil(ymax - ymin)))

    xmin = int(math.floor(xmin))
    ymin = int(math.floor(ymin))

    if width >= img_width:
        xmin = 0
        xmax = img_width
    else:
        xmax = xmin + width
        if xmin < 0:
            xmax -= xmin
            xmin = 0
        if xmax > img_width:
            xmin -= xmax - img_width
            xmax = img_width

    if height >= img_height:
        ymin = 0
        ymax = img_height
    else:
        ymax = ymin + height
        if ymin < 0:
            ymax -= ymin
            ymin = 0
        if ymax > img_height:
            ymin -= ymax - img_height
            ymax = img_height

    return int(xmin), int(ymin), int(xmax), int(ymax)


def _broaden_rect_bbox(xmin, ymin, xmax, ymax, img_width, img_height):
    """Return a detector-style rectangle that contains the input bbox."""
    width = max(1, xmax - xmin)
    height = max(1, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    roll = random.random()
    if roll < 0.20:
        mode = "none"
        new_xmin, new_ymin, new_xmax, new_ymax = xmin, ymin, xmax, ymax
    elif roll < 0.55:
        mode = "side_expand"
        max_side_expand = 0.75
        new_xmin = xmin - random.uniform(0.0, max_side_expand) * width
        new_xmax = xmax + random.uniform(0.0, max_side_expand) * width
        new_ymin = ymin - random.uniform(0.0, max_side_expand) * height
        new_ymax = ymax + random.uniform(0.0, max_side_expand) * height
    elif roll < 0.80:
        mode = "area_expand"
        scale = math.sqrt(random.uniform(1.0, 4.0))
        new_width = width * scale
        new_height = height * scale
        new_xmin = cx - new_width / 2.0
        new_xmax = cx + new_width / 2.0
        new_ymin = cy - new_height / 2.0
        new_ymax = cy + new_height / 2.0
    else:
        mode = "aspect_expand"
        target_aspect = random.uniform(0.35, 2.85)
        current_aspect = width / float(height)
        if target_aspect > current_aspect:
            new_width = height * target_aspect
            new_height = height
        else:
            new_width = width
            new_height = width / target_aspect
        new_xmin = cx - new_width / 2.0
        new_xmax = cx + new_width / 2.0
        new_ymin = cy - new_height / 2.0
        new_ymax = cy + new_height / 2.0

    new_xmin, new_ymin, new_xmax, new_ymax = _fit_rect_to_image(
        new_xmin, new_ymin, new_xmax, new_ymax, img_width, img_height
    )
    return new_xmin, new_ymin, new_xmax, new_ymax, mode


def _normalise_crop_coordinates(crop_coordinates):
    if crop_coordinates is None:
        return None, None
    if len(crop_coordinates) >= 4 and isinstance(crop_coordinates[3], dict):
        return crop_coordinates[:3], crop_coordinates[3]
    return crop_coordinates, None


def _make_crop_state(processed_bboxes, idx_bbox_ref):
    return {
        "idx_bbox_ref": int(idx_bbox_ref),
        "processed_bboxes": [dict(cur_bbox) for cur_bbox in processed_bboxes],
    }


def _interval_containing_required(
    required_min,
    required_max,
    desired_center,
    side,
    limit_min,
    limit_max,
):
    required_min = int(required_min)
    required_max = int(required_max)
    side = max(int(side), required_max - required_min, 1)
    low = max(int(limit_min), required_max - side)
    high = min(int(required_min), int(limit_max) - side)
    if low > high:
        low = required_max - side
        high = required_min
    start = int(round(float(desired_center) - side / 2.0))
    start = min(max(start, low), high)
    return start, start + side


def _square_bbox_containing_required(
    required_bbox,
    desired_bbox,
    side,
    width,
    height,
    border=0,
):
    req_xmin, req_ymin, req_xmax, req_ymax = required_bbox
    des_xmin, des_ymin, des_xmax, des_ymax = desired_bbox
    side = max(
        int(side),
        int(req_xmax) - int(req_xmin),
        int(req_ymax) - int(req_ymin),
        1,
    )
    desired_cx = (float(des_xmin) + float(des_xmax)) / 2.0
    desired_cy = (float(des_ymin) + float(des_ymax)) / 2.0
    xmin, xmax = _interval_containing_required(
        req_xmin, req_xmax, desired_cx, side, border, width - border
    )
    ymin, ymax = _interval_containing_required(
        req_ymin, req_ymax, desired_cy, side, border, height - border
    )
    return xmin, ymin, xmax, ymax


def _shift_processed_bbox(cur_bbox, dx, dy):
    for x_key in ["xmin", "xmax", "original_xmin", "original_xmax"]:
        if x_key in cur_bbox:
            cur_bbox[x_key] += dx
    for y_key in ["ymin", "ymax", "original_ymin", "original_ymax"]:
        if y_key in cur_bbox:
            cur_bbox[y_key] += dy


def crop_image(
    img_path,
    bbox_path,
    mask_random_offset,
    mask_delta,
    crop_delta,
    mask_square,
    crop_dim,
    output_dim,
    context_pixels,
    load_size,
    load_size_keep_ratio=False,
    get_crop_coordinates=False,
    crop_coordinates=None,
    select_cat=-1,
    crop_center=False,
    fixed_mask_size=-1,
    fixed_mask_size_model=-1,
    fixed_mask_min_unmasked_border_model=4,
    bbox_ref_id=-1,
    inverted_mask=False,
    single_bbox=False,
    override_class=-1,
    min_crop_bbox_ratio=None,
    random_bbox=False,
    return_meta=False,
    broaden_rect_aug=False,
):
    margin = context_pixels * 2
    x_padding = 0
    y_padding = 0

    try:
        img = load_image(img_path)
        old_size = img.size
        resize_scale = 1.0
        effective_crop_dim = crop_dim
        effective_crop_delta = crop_delta
        effective_fixed_mask_size = fixed_mask_size
        effective_mask_delta = mask_delta

        if load_size != []:
            target_width = int(load_size[0])
            target_height = int(load_size[1] if len(load_size) > 1 else load_size[0])
            if target_width <= 0 or target_height <= 0:
                raise ValueError(f"load_size must contain positive values: {load_size}")

            if load_size_keep_ratio:
                target_long_side = max(target_width, target_height)
                source_long_side = max(old_size)
                resize_scale = target_long_side / float(source_long_side)
                new_width = max(1, int(round(old_size[0] * resize_scale)))
                new_height = max(1, int(round(old_size[1] * resize_scale)))
            else:
                new_width = target_width
                new_height = target_height

            old_size = img.size
            img = F.resize(img, (new_height, new_width))
            ratio_x = img.size[0] / old_size[0]
            ratio_y = img.size[1] / old_size[1]
        else:
            ratio_x = 1
            ratio_y = 1

        if load_size_keep_ratio and load_size != []:
            effective_crop_dim = max(1, int(round(crop_dim * resize_scale)))
            effective_crop_delta = max(0, int(round(crop_delta * resize_scale)))
            if fixed_mask_size > 0:
                effective_fixed_mask_size = max(
                    1, int(round(fixed_mask_size * resize_scale))
                )
            effective_mask_delta = _scale_pixel_mask_delta(mask_delta, resize_scale)

        img = np.array(img)
        loaded_img_height, loaded_img_width = img.shape[:2]
    except Exception as e:
        raise ValueError(f"failure with loading image {img_path}") from e

    try:
        if bbox_path is not None and bbox_path.endswith(".txt"):
            # Bbox file
            f = open(bbox_path, "r")
        elif random_bbox:
            bbox_path = ""
        else:
            import cv2

            bbox_img = cv2.imread(bbox_path)

    except Exception as e:
        raise ValueError(
            f"failure with loading label {bbox_path} for image {img_path}"
        ) from e

    if bbox_path.endswith(".txt"):
        # Bbox file

        with f:
            bboxes = []
            for line in f:
                if len(line) > 2:  # to make sure the current line is a real bbox
                    if override_class != -1:
                        bbox = line.split()
                        bbox[0] = str(override_class)
                        line = " ".join(bbox)
                        bboxes.append(line)
                    elif select_cat != -1:
                        bbox = line.split()
                        cat = int(bbox[0])
                        if cat != select_cat:
                            continue  # skip bboxes
                        else:
                            bboxes.append(line)
                    else:
                        bboxes.append(line)
                elif line != "" or line != " ":
                    print("%s does not describe a bbox" % line)

    elif random_bbox:
        bboxes = []
        xmin = np.random.randint(0, old_size[0] - 1)
        ymin = np.random.randint(0, old_size[1] - 1)
        xmax = np.random.randint(
            xmin, min(xmin + crop_dim, old_size[0])
        )  # min(xmin+crop_dim,img.shape[1]))
        ymax = np.random.randint(
            ymin, min(ymin + crop_dim, old_size[1])
        )  # min(ymin+crop_dim,img.shape[0]))
        bboxes.append(f"1 {xmin} {ymin} {xmax} {ymax}")

    else:
        cat = str(int(np.max(bbox_img)))

        # Find the indices of non-zero elements in the image
        non_zero_indices = np.nonzero(bbox_img)

        # Extract the rows and columns containing non-zero elements
        non_zero_rows = non_zero_indices[1]
        non_zero_cols = non_zero_indices[0]

        # Find the minimum and maximum indices for rows and columns
        min_row, max_row = np.min(non_zero_rows), np.max(non_zero_rows)
        min_col, max_col = np.min(non_zero_cols), np.max(non_zero_cols)

        xmin = str(min_row)
        ymin = str(min_col)
        xmax = str(max_row)
        ymax = str(max_col)

        bboxes = [cat + " " + xmin + " " + ymin + " " + xmax + " " + ymax]

    crop_coordinates, crop_state = _normalise_crop_coordinates(crop_coordinates)

    # If one bbox only per crop
    if single_bbox and bbox_ref_id == -1:
        bbox_ref_id = np.random.randint(low=0, high=len(bboxes))

    # If a bbox ref is given, only select for that box
    if bbox_ref_id >= 0:
        bboxes_tmp = []
        bboxes_tmp.append(bboxes[bbox_ref_id])
        bboxes = bboxes_tmp

    if len(bboxes) == 0:
        raise ValueError(f"There is no bbox at {bbox_path} for image {img_path}.")

    # Creation of a blank mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    processed_bboxes = []
    square_model_border_active = (
        bool(mask_square)
        and fixed_mask_size_model <= 0
        and fixed_mask_min_unmasked_border_model > 0
    )

    # A bbox of reference will be used to compute the crop
    if crop_state is not None and "idx_bbox_ref" in crop_state:
        idx_bbox_ref = crop_state["idx_bbox_ref"]
    else:
        idx_bbox_ref = random.randint(0, len(bboxes) - 1)

    processed_bbox_state = {}
    if crop_state is not None:
        for cur_bbox in crop_state.get("processed_bboxes", []):
            processed_bbox_state[cur_bbox["index"]] = cur_bbox

    for i, cur_bbox in enumerate(bboxes):
        bbox = cur_bbox.split()
        cat = int(bbox[0])
        if select_cat != -1:
            if cat != select_cat:
                continue  # skip bboxes

        xmin = math.floor(int(bbox[1]) * ratio_x)
        ymin = math.floor(int(bbox[2]) * ratio_y)
        xmax = math.floor(int(bbox[3]) * ratio_x)
        ymax = math.floor(int(bbox[4]) * ratio_y)

        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        original_xmin, original_ymin, original_xmax, original_ymax = (
            xmin,
            ymin,
            xmax,
            ymax,
        )

        if i in processed_bbox_state:
            state_bbox = processed_bbox_state[i]
            xmin = state_bbox["xmin"]
            ymin = state_bbox["ymin"]
            xmax = state_bbox["xmax"]
            ymax = state_bbox["ymax"]
            augment_mode = state_bbox.get("augment_mode", "reused")
        else:
            augment_mode = "none"

            if effective_mask_delta != [[]]:
                if len(effective_mask_delta) == 1:
                    if isinstance(effective_mask_delta[0][0], float):
                        if len(effective_mask_delta[0]) == 1:
                            mask_delta_x = effective_mask_delta[0][0] * bbox_width
                            mask_delta_y = effective_mask_delta[0][0] * bbox_height
                        else:
                            mask_delta_x = effective_mask_delta[0][0] * bbox_width
                            mask_delta_y = effective_mask_delta[0][1] * bbox_height
                    elif isinstance(effective_mask_delta[0][0], int):
                        if len(effective_mask_delta[0]) == 1:
                            mask_delta_x = effective_mask_delta[0][0]
                            mask_delta_y = effective_mask_delta[0][0]
                        else:
                            mask_delta_x = effective_mask_delta[0][0]
                            mask_delta_y = effective_mask_delta[0][1]
                    else:
                        raise ValueError("mask_delta value is incorrect.")
                else:
                    if len(effective_mask_delta) <= cat - 1:
                        raise ValueError("too few classes, can't find mask_delta value")
                    mask_delta_cat = effective_mask_delta[cat - 1]
                    if isinstance(effective_mask_delta[0][0], float):
                        if len(mask_delta_cat) == 1:
                            mask_delta_x = mask_delta_cat[0] * bbox_width
                            mask_delta_y = mask_delta_cat[0] * bbox_height
                        else:
                            mask_delta_x = mask_delta_cat[0] * bbox_width
                            mask_delta_y = mask_delta_cat[1] * bbox_height
                    elif isinstance(effective_mask_delta[0][0], int):
                        if len(mask_delta_cat) == 1:
                            mask_delta_x = mask_delta_cat[0]
                            mask_delta_y = mask_delta_cat[0]
                        else:
                            mask_delta_x = mask_delta_cat[0]
                            mask_delta_y = mask_delta_cat[1]
                    else:
                        raise ValueError("mask_delta value is incorrect.")

                mask_delta_x = int(mask_delta_x)
                mask_delta_y = int(mask_delta_y)

                if mask_delta_x > 0 or mask_delta_y > 0:
                    ymin -= mask_delta_y
                    ymax += mask_delta_y
                    xmin -= mask_delta_x
                    xmax += mask_delta_x

            if len(mask_random_offset) == 1:
                mask_random_offset_x = mask_random_offset[0]
                mask_random_offset_y = mask_random_offset[0]
            elif len(mask_random_offset) == 2:
                mask_random_offset_x = mask_random_offset[0]
                mask_random_offset_y = mask_random_offset[1]

            # from ratio to pixel gap
            mask_random_offset_x = round(mask_random_offset_x * (xmax - xmin))
            mask_random_offset_y = round(mask_random_offset_y * (ymax - ymin))

            if mask_random_offset_x > 0 or mask_random_offset_y > 0:
                ymin -= random.randint(0, mask_random_offset_y)
                ymax += random.randint(0, mask_random_offset_y)
                xmin -= random.randint(0, mask_random_offset_x)
                xmax += random.randint(0, mask_random_offset_x)

            if broaden_rect_aug:
                xmin, ymin, xmax, ymax, augment_mode = _broaden_rect_bbox(
                    xmin, ymin, xmax, ymax, img.shape[1], img.shape[0]
                )

            if mask_square:
                sdiff = (xmax - xmin) - (ymax - ymin)
                if sdiff > 0:
                    ymax += int(sdiff / 2)
                    ymin -= int(sdiff / 2)
                else:
                    xmax += -int(sdiff / 2)
                    xmin -= -int(sdiff / 2)

            if effective_fixed_mask_size > 0:
                xdiff = effective_fixed_mask_size - (xmax - xmin)
                ydiff = effective_fixed_mask_size - (ymax - ymin)

                ymax += int(ydiff / 2)
                ymin -= int(ydiff / 2)

                xmax += int(xdiff / 2)
                xmin -= int(xdiff / 2)

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, img.shape[1])
        ymax = min(ymax, img.shape[0])

        mask[ymin:ymax, xmin:xmax] = np.full((ymax - ymin, xmax - xmin), cat)
        processed_bboxes.append(
            {
                "index": i,
                "cat": cat,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "original_xmin": original_xmin,
                "original_ymin": original_ymin,
                "original_xmax": original_xmax,
                "original_ymax": original_ymax,
                "augment_mode": augment_mode,
            }
        )

        if i == idx_bbox_ref:
            cat_ref = cat
            x_min_ref = xmin
            x_max_ref = xmax
            y_min_ref = ymin
            y_max_ref = ymax

            if (
                x_min_ref < context_pixels
                or y_min_ref < context_pixels
                or x_max_ref + context_pixels > img.shape[1]
                or y_max_ref + context_pixels > img.shape[0]
            ):
                new_context_pixels = min(
                    x_min_ref,
                    y_min_ref,
                    img.shape[1] - x_max_ref,
                    img.shape[0] - y_max_ref,
                )

                warnings.warn(
                    f"Bbox is too close to the edge to crop with context ({context_pixels} pixels)  for {img_path},using context_pixels=distance to the edge {new_context_pixels}"
                )

                context_pixels = new_context_pixels

    height = y_max_ref - y_min_ref
    width = x_max_ref - x_min_ref
    source_img_width = None
    source_img_height = None

    def clipped_original_bbox(cur_bbox):
        if source_img_width is not None and source_img_height is not None:
            width_limit = source_img_width
            height_limit = source_img_height
        elif hasattr(img, "shape"):
            height_limit, width_limit = img.shape[:2]
        else:
            width_limit, height_limit = img.size
        return (
            max(0, min(cur_bbox["original_xmin"], width_limit)),
            max(0, min(cur_bbox["original_ymin"], height_limit)),
            max(0, min(cur_bbox["original_xmax"], width_limit)),
            max(0, min(cur_bbox["original_ymax"], height_limit)),
        )

    def square_model_border_params(crop_size):
        output_side = output_dim + margin
        border = int(fixed_mask_min_unmasked_border_model)
        max_model_mask_side = output_side - 2 * border
        if max_model_mask_side < 1:
            raise ValueError(
                f"square model mask border {border} is too large for output size {output_side}"
            )
        max_source_side = int(math.floor(max_model_mask_side * crop_size / output_side))
        return output_side, border, max_model_mask_side, max(1, max_source_side)

    def ref_original_side():
        for cur_bbox in processed_bboxes:
            if cur_bbox["index"] == idx_bbox_ref:
                oxmin, oymin, oxmax, oymax = clipped_original_bbox(cur_bbox)
                return max(oxmax - oxmin, oymax - oymin)
        return max(width, height)

    def apply_square_model_border_mask(crop_size):
        nonlocal mask
        nonlocal x_min_ref, x_max_ref, y_min_ref, y_max_ref

        if not square_model_border_active:
            return

        _, _, _, max_source_side = square_model_border_params(crop_size)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for cur_bbox in processed_bboxes:
            oxmin, oymin, oxmax, oymax = clipped_original_bbox(cur_bbox)
            candidate_side = max(
                cur_bbox["xmax"] - cur_bbox["xmin"],
                cur_bbox["ymax"] - cur_bbox["ymin"],
            )
            original_side = max(oxmax - oxmin, oymax - oymin)
            side = max(original_side, min(candidate_side, max_source_side))
            xmin, ymin, xmax, ymax = _square_bbox_containing_required(
                (oxmin, oymin, oxmax, oymax),
                (
                    cur_bbox["xmin"],
                    cur_bbox["ymin"],
                    cur_bbox["xmax"],
                    cur_bbox["ymax"],
                ),
                side,
                img.shape[1],
                img.shape[0],
            )
            cur_bbox.update({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
            mask[ymin:ymax, xmin:xmax] = np.full((ymax - ymin, xmax - xmin), cur_bbox["cat"])
            if cur_bbox["index"] == idx_bbox_ref:
                x_min_ref = xmin
                x_max_ref = xmax
                y_min_ref = ymin
                y_max_ref = ymax

    def pad_image_and_mask(left=0, right=0, top=0, bottom=0):
        nonlocal img, mask
        nonlocal x_padding, y_padding
        nonlocal x_min_ref, x_max_ref, y_min_ref, y_max_ref

        left = int(max(0, left))
        right = int(max(0, right))
        top = int(max(0, top))
        bottom = int(max(0, bottom))
        if left == 0 and right == 0 and top == 0 and bottom == 0:
            return

        img = np.pad(
            img,
            ((top, bottom), (left, right), (0, 0)),
            "constant",
            constant_values=0,
        )
        mask = np.pad(
            mask,
            ((top, bottom), (left, right)),
            "constant",
            constant_values=0,
        )

        x_padding += left
        y_padding += top
        x_max_ref += left
        x_min_ref += left
        y_max_ref += top
        y_min_ref += top
        for cur_bbox in processed_bboxes:
            _shift_processed_bbox(cur_bbox, left, top)

    def ensure_crop_canvas(crop_size, source_border=0):
        left = max(0, context_pixels + source_border - x_min_ref)
        right = max(0, x_max_ref + source_border + context_pixels - img.shape[1])
        top = max(0, context_pixels + source_border - y_min_ref)
        bottom = max(0, y_max_ref + source_border + context_pixels - img.shape[0])

        min_side = crop_size + 2 * context_pixels
        extra_w = max(0, min_side - (img.shape[1] + left + right))
        extra_h = max(0, min_side - (img.shape[0] + top + bottom))
        left += int(math.ceil(extra_w / 2.0))
        right += int(math.floor(extra_w / 2.0))
        top += int(math.ceil(extra_h / 2.0))
        bottom += int(math.floor(extra_h / 2.0))
        pad_image_and_mask(left=left, right=right, top=top, bottom=bottom)

    def source_border_for_crop(crop_size):
        if not square_model_border_active:
            return 0
        output_side, border, _, _ = square_model_border_params(crop_size)
        return int(math.ceil(border * crop_size / output_side))

    def ensure_crop_size_for_square_border(crop_size):
        if not square_model_border_active:
            return crop_size

        crop_size = int(crop_size)
        for _ in range(16):
            apply_square_model_border_mask(crop_size)
            source_border = source_border_for_crop(crop_size)
            required_size = max(
                x_max_ref - x_min_ref + 2 * source_border,
                y_max_ref - y_min_ref + 2 * source_border,
                1,
            )
            if required_size <= crop_size:
                return crop_size
            crop_size = int(required_size)
        raise ValueError(f"Crop size cannot be computed for {img_path}")

    def apply_fixed_model_mask(crop_size):
        nonlocal mask
        nonlocal x_min_ref, x_max_ref, y_min_ref, y_max_ref

        if fixed_mask_size_model <= 0:
            return

        output_side = output_dim + margin
        max_model_mask_side = output_side - 2 * fixed_mask_min_unmasked_border_model
        if max_model_mask_side < 1:
            raise ValueError(
                f"fixed model mask border {fixed_mask_min_unmasked_border_model} is too large for output size {output_side}"
            )
        if fixed_mask_size_model > output_side:
            raise ValueError(
                f"fixed model mask size {fixed_mask_size_model} is larger than output size {output_side}"
            )
        fixed_model_side = min(fixed_mask_size_model, max_model_mask_side)
        fixed_source_side = int(round(fixed_model_side * crop_size / output_side))
        fixed_source_side = max(1, fixed_source_side)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for cur_bbox in processed_bboxes:
            side = max(
                fixed_source_side,
                cur_bbox["xmax"] - cur_bbox["xmin"],
                cur_bbox["ymax"] - cur_bbox["ymin"],
            )
            cx = (cur_bbox["xmin"] + cur_bbox["xmax"]) / 2.0
            cy = (cur_bbox["ymin"] + cur_bbox["ymax"]) / 2.0
            xmin = int(round(cx - side / 2.0))
            ymin = int(round(cy - side / 2.0))
            xmax = xmin + side
            ymax = ymin + side
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(xmax, img.shape[1])
            ymax = min(ymax, img.shape[0])
            mask[ymin:ymax, xmin:xmax] = np.full(
                (ymax - ymin, xmax - xmin), cur_bbox["cat"]
            )
            cur_bbox.update({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
            if cur_bbox["index"] == idx_bbox_ref:
                x_min_ref = xmin
                x_max_ref = xmax
                y_min_ref = ymin
                y_max_ref = ymax

    # Let's compute crop size
    if crop_coordinates is None:
        # We compute the range within which crop size should be

        # Crop size should be > height, width bbox (to keep the bbox within the crop)
        # Crop size should be > crop_dim - delta
        if square_model_border_active:
            required_bbox_side = ref_original_side()
        else:
            required_bbox_side = max(height, width)
        crop_size_min = max(required_bbox_side, effective_crop_dim - effective_crop_delta)

        # Crop size should be < crop_dim - delta
        crop_size_max = effective_crop_dim + effective_crop_delta

        # if bbox is bigger than crop size min, we replace crop size min by bbox
        # So, we'll have bbox size <= crop size <= crop size max

        # if bbox is bigger than crop size max, we replace crop size max by bbox
        # So, we'll have bbox size == crop size

        if crop_size_max < required_bbox_side:
            crop_size_max = required_bbox_side
            warnings.warn(
                f"Bbox size ({height}, {width}) > crop dim {crop_size_max} for {img_path}, using crop_dim = bbox size"
            )

        if crop_size_max < crop_size_min:
            raise ValueError(f"Crop size cannot be computed for {img_path}")

        if min_crop_bbox_ratio:
            expected_crop_size = round(max(height, width) * min_crop_bbox_ratio)
            if crop_size_max < expected_crop_size:
                warnings.warn(f"Enlarging crop to match min_crop_bbox_ratio")
                crop_size_min = expected_crop_size
                crop_size_max = expected_crop_size

        if square_model_border_active:
            output_side, _, max_model_mask_side, _ = square_model_border_params(
                max(1, crop_size_min)
            )
            required_crop_size = int(
                math.ceil(ref_original_side() * output_side / max_model_mask_side)
            )
            if required_crop_size > crop_size_min:
                crop_size_min = required_crop_size
            if crop_size_max < crop_size_min:
                warnings.warn(
                    "Enlarging crop to satisfy data_online_creation_mask_min_unmasked_border with square masks"
                )
                crop_size_max = crop_size_min

        crop_size = random.randint(crop_size_min, crop_size_max)
        apply_fixed_model_mask(crop_size)
        crop_size = ensure_crop_size_for_square_border(crop_size)

        if crop_size > min(img.shape[0], img.shape[1]):
            warnings.warn(
                f"Image size ({img.shape}) < crop dim {crop_size} for {img_path}, zero padding is done on image"
            )

            if crop_size > img.shape[0]:
                pad_y = crop_size - img.shape[0]
            else:
                pad_y = 0

            if crop_size > img.shape[1]:
                pad_x = crop_size - img.shape[1]
            else:
                pad_x = 0

            pad_x = math.ceil(pad_x / 2)
            pad_y = math.ceil(pad_y / 2)
            pad_image_and_mask(
                left=pad_x,
                right=pad_x,
                top=pad_y,
                bottom=pad_y,
            )

        # Let's compute crop position
        # The final crop coordinates will be [x_crop:x_crop+crop_size+margin,y_crop:y_crop+crop_size+margin)

        # The bbox needs to fit in the crop without context
        # So x_crop <= x_min_ref and x_crop + crop_size >= x_max_ref

        # The crop with context needs to fit in the img
        # So x_crop - context_pixels >=0 and x_crop + crop_size + context_pixels <= img_size-1

        if square_model_border_active:
            source_border = source_border_for_crop(crop_size)
        else:
            source_border = 0
        ensure_crop_canvas(crop_size, source_border)

        x_crop_min = max(context_pixels, x_max_ref + source_border - crop_size)
        x_crop_max = min(
            x_min_ref - source_border, img.shape[1] - crop_size - context_pixels
        )

        y_crop_min = max(context_pixels, y_max_ref + source_border - crop_size)
        y_crop_max = min(
            y_min_ref - source_border, img.shape[0] - crop_size - context_pixels
        )

        if x_crop_min > x_crop_max or y_crop_min > y_crop_max:
            raise ValueError(f"Crop position cannot be computed for {img_path}")

        if crop_center:
            x_crop = (x_crop_min + x_crop_max) // 2
            y_crop = (y_crop_min + y_crop_max) // 2
        else:
            x_crop = random.randint(x_crop_min, x_crop_max)
            y_crop = random.randint(y_crop_min, y_crop_max)

        if (
            x_crop < context_pixels
            or x_crop + crop_size + context_pixels > img.shape[1]
            or y_crop < context_pixels
            or y_crop + crop_size + context_pixels > img.shape[0]
        ):
            raise ValueError(
                f"Image cropping failed for {img_path}.",
            )

        if get_crop_coordinates:
            if broaden_rect_aug or crop_state is not None:
                return (
                    x_crop - x_min_ref,
                    y_crop - y_min_ref,
                    crop_size,
                    _make_crop_state(processed_bboxes, idx_bbox_ref),
                )
            return x_crop - x_min_ref, y_crop - y_min_ref, crop_size

    else:
        x_crop, y_crop, crop_size = crop_coordinates
        apply_fixed_model_mask(crop_size)
        crop_size = ensure_crop_size_for_square_border(crop_size)
        x_crop = x_crop + x_min_ref
        y_crop = y_crop + y_min_ref

        if x_crop < context_pixels:
            x_crop = context_pixels

        if x_crop + crop_size + context_pixels > img.shape[1]:
            x_crop = img.shape[1] - (crop_size + context_pixels)

        if y_crop < context_pixels:
            y_crop = context_pixels

        if y_crop + crop_size + context_pixels > img.shape[0]:
            y_crop = img.shape[0] - (crop_size + context_pixels)

        if square_model_border_active:
            source_border = source_border_for_crop(crop_size)
            ensure_crop_canvas(crop_size, source_border)
            x_crop_min = max(context_pixels, x_max_ref + source_border - crop_size)
            x_crop_max = min(
                x_min_ref - source_border, img.shape[1] - crop_size - context_pixels
            )
            y_crop_min = max(context_pixels, y_max_ref + source_border - crop_size)
            y_crop_max = min(
                y_min_ref - source_border, img.shape[0] - crop_size - context_pixels
            )
            if x_crop_min > x_crop_max or y_crop_min > y_crop_max:
                raise ValueError(f"Crop position cannot be computed for {img_path}")
            x_crop = min(max(x_crop, x_crop_min), x_crop_max)
            y_crop = min(max(y_crop, y_crop_min), y_crop_max)

    source_img_height, source_img_width = img.shape[:2]
    img = img[
        y_crop - context_pixels : y_crop + crop_size + context_pixels,
        x_crop - context_pixels : x_crop + crop_size + context_pixels,
        :,
    ]

    img = Image.fromarray(img)

    img = F.resize(img, output_dim + margin)

    mask = mask[
        y_crop : y_crop + crop_size + margin,
        x_crop : x_crop + crop_size + margin,
    ]

    x_max_ref -= x_crop
    x_min_ref -= x_crop
    y_max_ref -= y_crop
    y_min_ref -= y_crop

    ref_bbox = [x_min_ref, y_min_ref, x_max_ref, y_max_ref]

    # invert mask if required
    if inverted_mask:
        mask[mask > 0] = 2
        mask[mask == 0] = 1
        mask[mask == 2] = 0

    if fixed_mask_size_model > 0:
        output_side = output_dim + margin
        border = fixed_mask_min_unmasked_border_model
        max_mask_side = output_side - 2 * border
        if max_mask_side < 1:
            raise ValueError(
                f"fixed model mask border {border} is too large for output size {output_side}"
            )
        resized_mask = np.zeros((output_side, output_side), dtype=np.uint8)
        for cur_bbox in processed_bboxes:
            xmin = cur_bbox["xmin"] - x_crop
            xmax = cur_bbox["xmax"] - x_crop
            ymin = cur_bbox["ymin"] - y_crop
            ymax = cur_bbox["ymax"] - y_crop
            xmin = int(round(xmin * output_side / crop_size))
            xmax = int(round(xmax * output_side / crop_size))
            ymin = int(round(ymin * output_side / crop_size))
            ymax = int(round(ymax * output_side / crop_size))

            side = max(xmax - xmin, ymax - ymin)
            side = max(side, fixed_mask_size_model)
            side = min(side, max_mask_side)
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            xmin = int(round(cx - side / 2.0))
            ymin = int(round(cy - side / 2.0))
            xmin = min(max(xmin, border), output_side - border - side)
            ymin = min(max(ymin, border), output_side - border - side)
            xmax = xmin + side
            ymax = ymin + side
            resized_mask[ymin:ymax, xmin:xmax] = cur_bbox["cat"]
        if inverted_mask:
            resized_mask[resized_mask > 0] = 2
            resized_mask[resized_mask == 0] = 1
            resized_mask[resized_mask == 2] = 0
        mask = Image.fromarray(resized_mask)
    elif square_model_border_active:
        output_side, border, max_mask_side, _ = square_model_border_params(crop_size)
        resized_mask = np.zeros((output_side, output_side), dtype=np.uint8)
        for cur_bbox in processed_bboxes:
            oxmin_src, oymin_src, oxmax_src, oymax_src = clipped_original_bbox(cur_bbox)
            xmin = cur_bbox["xmin"] - x_crop
            xmax = cur_bbox["xmax"] - x_crop
            ymin = cur_bbox["ymin"] - y_crop
            ymax = cur_bbox["ymax"] - y_crop
            oxmin = oxmin_src - x_crop
            oxmax = oxmax_src - x_crop
            oymin = oymin_src - y_crop
            oymax = oymax_src - y_crop

            xmin = int(round(xmin * output_side / crop_size))
            xmax = int(round(xmax * output_side / crop_size))
            ymin = int(round(ymin * output_side / crop_size))
            ymax = int(round(ymax * output_side / crop_size))
            oxmin = int(round(oxmin * output_side / crop_size))
            oxmax = int(round(oxmax * output_side / crop_size))
            oymin = int(round(oymin * output_side / crop_size))
            oymax = int(round(oymax * output_side / crop_size))

            side = max(xmax - xmin, ymax - ymin)
            original_side = max(oxmax - oxmin, oymax - oymin)
            side = max(original_side, min(side, max_mask_side))
            xmin, ymin, xmax, ymax = _square_bbox_containing_required(
                (oxmin, oymin, oxmax, oymax),
                (xmin, ymin, xmax, ymax),
                side,
                output_side,
                output_side,
                border=border,
            )
            resized_mask[ymin:ymax, xmin:xmax] = cur_bbox["cat"]
        if inverted_mask:
            resized_mask[resized_mask > 0] = 2
            resized_mask[resized_mask == 0] = 1
            resized_mask[resized_mask == 2] = 0
        mask = Image.fromarray(resized_mask)
    else:
        mask = Image.fromarray(mask)
        mask = F.resize(
            mask, output_dim + margin, interpolation=InterpolationMode.NEAREST
        )

    # resize ref_bbox to output_dim + margin
    ref_bbox = [
        cat_ref,
        int(ref_bbox[0] * (output_dim + margin) / crop_size),
        int(ref_bbox[1] * (output_dim + margin) / crop_size),
        int(ref_bbox[2] * (output_dim + margin) / crop_size),
        int(ref_bbox[3] * (output_dim + margin) / crop_size),
    ]

    if return_meta:
        crop_meta = {
            "orig_width": int(old_size[0]),
            "orig_height": int(old_size[1]),
            "loaded_width": int(loaded_img_width),
            "loaded_height": int(loaded_img_height),
            "x_padding": int(x_padding),
            "y_padding": int(y_padding),
            "x_crop": int(x_crop),
            "y_crop": int(y_crop),
            "crop_size": int(crop_size),
            "context_pixels": int(context_pixels),
            "mask_broaden_rect_aug": bool(broaden_rect_aug),
            "processed_bboxes": [dict(cur_bbox) for cur_bbox in processed_bboxes],
        }
        return img, mask, ref_bbox, idx_bbox_ref, crop_meta

    return img, mask, ref_bbox, idx_bbox_ref


def fill_mask_with_random(img, mask, cls):
    """
    Randomize image inside masks.
    cls: class to replace by random noise, if -1 all classes are replaced
    """
    if cls == -1:
        mask = torch.where(mask != 0, 1.0, 0.0)
    else:
        mask = torch.where(mask == cls, 1.0, 0.0)
    noise = torch.randn_like(img)
    return img * (1 - mask) + noise * mask


def fill_mask_with_color(img, mask, colors):
    """
    Fill image with color at the place
    colors: dict with tuple (r, g, b) between -1 and 1
    """
    all_cls = mask.unique()

    for cls in all_cls:
        if cls == 0:
            continue
        if cls in colors:
            color = colors[cls]
        else:
            color = (0, 0, 0)
        mask = torch.where(mask == cls, 1.0, 0.0)
        dims = img.shape

        assert (
            len(color) == dims[-3]
        ), "fill_mask_with_color: number of channels does not match"
        color = torch.tensor(color).repeat_interleave(dims[-2] * dims[-1]).reshape(dims)
        img = img * (1 - mask) + color * mask

    return img


def sanitize_paths(
    paths_img,
    paths_bb,
    mask_random_offset,
    mask_delta,
    crop_delta,
    mask_square,
    crop_dim,
    output_dim,
    context_pixels,
    load_size,
    load_size_keep_ratio=False,
    fixed_mask_size_model=-1,
    fixed_mask_min_unmasked_border_model=4,
    broaden_rect_aug=False,
    select_cat=-1,
    data_relative_paths=False,
    data_root_path="",
    max_dataset_size=float("inf"),
    verbose=False,
):
    return_paths_img = []
    return_paths_bb = []

    if paths_bb is None:
        paths_bb = [None for k in range(len(paths_img))]

    for path_img, path_bb in tqdm(zip(paths_img, paths_bb)):
        if data_relative_paths:
            path_img = os.path.join(data_root_path, path_img)
            path_bb = os.path.join(data_root_path, path_bb)

        if len(return_paths_img) >= max_dataset_size:
            break

        failed = False
        try:
            load_img(path_img)
            if path_bb is not None:
                try:
                    crop_image(
                        path_img,
                        path_bb,
                        mask_random_offset=mask_random_offset,
                        mask_delta=mask_delta,
                        crop_delta=0,
                        mask_square=mask_square,
                        fixed_mask_size_model=fixed_mask_size_model,
                        fixed_mask_min_unmasked_border_model=fixed_mask_min_unmasked_border_model,
                        broaden_rect_aug=broaden_rect_aug,
                        crop_dim=crop_dim + crop_delta,
                        output_dim=output_dim,
                        context_pixels=context_pixels,
                        load_size=load_size,
                        load_size_keep_ratio=load_size_keep_ratio,
                        select_cat=select_cat,
                    )
                except Exception as e:
                    failed = True
                    error = e
        except Exception as e:
            failed = True
            error = e

        if failed:
            if verbose:
                print("failed", path_img, path_bb)
                print(error)
        else:
            return_paths_img.append(path_img)
            return_paths_bb.append(path_bb)

    print(
        "%d images deleted over %d,remaining %d images"
        % (
            len(paths_img) - len(return_paths_img),
            len(paths_img),
            len(return_paths_img),
        )
    )

    return return_paths_img, return_paths_bb


def write_paths_file(img_paths, label_paths, file_path):
    try:
        with open(file_path, "w") as f:
            for img_path, label_path in zip(img_paths, label_paths):
                if label_path is None:
                    label_path = ""
                cur_line = img_path + " " + label_path
                f.write(cur_line)
                f.write("\n")
    except Exception as e:
        print("failed saving sanitized paths file at ", file_path)
        print(e)

    print("sanitized paths file saved at ", file_path)
