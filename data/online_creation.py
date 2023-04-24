import math
import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import warnings
import os


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
    get_crop_coordinates=False,
    crop_coordinates=None,
    select_cat=-1,
    crop_center=False,
    fixed_mask_size=-1,
    bbox_ref_id=-1,
    inverted_mask=False,
    single_bbox=False,
    override_class=-1,
):

    margin = context_pixels * 2

    try:
        img = Image.open(img_path).convert("RGB")
        if load_size != []:
            old_size = img.size
            img = F.resize(img, load_size)
            new_size = img.size
            ratio_x = img.size[0] / old_size[0]
            ratio_y = img.size[1] / old_size[1]
        else:
            ratio_x = 1
            ratio_y = 1

        img = np.array(img)
    except Exception as e:
        raise ValueError(f"failure with loading image {img_path}") from e

    try:
        f = open(bbox_path, "r")
    except Exception as e:
        raise ValueError(
            f"failure with loading label {bbox_path} for image {img_path}"
        ) from e

    with f:
        bboxes = []

        for line in f:
            if len(line) > 2:  # to make sure the current line is a real bbox
                if override_class != -1:
                    bbox = line.split()
                    bbox[0] = override_class
                    line = " ".join(bbox)

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

        # A bbox of reference will be used to compute the crop
        idx_bbox_ref = random.randint(0, len(bboxes) - 1)

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

            if len(mask_delta) == 1:
                mask_delta_x = mask_delta[0]
                mask_delta_y = mask_delta[0]
            elif len(mask_delta) == 2:
                mask_delta_x = mask_delta[0]
                mask_delta_y = mask_delta[1]

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

            if mask_square:
                sdiff = (xmax - xmin) - (ymax - ymin)
                if sdiff > 0:
                    ymax += int(sdiff / 2)
                    ymin -= int(sdiff / 2)
                else:
                    xmax += -int(sdiff / 2)
                    xmin -= -int(sdiff / 2)

            if fixed_mask_size > 0:
                xdiff = fixed_mask_size - (xmax - xmin)
                ydiff = fixed_mask_size - (ymax - ymin)

                ymax += int(ydiff / 2)
                ymin -= int(ydiff / 2)

                xmax += int(xdiff / 2)
                xmin -= int(xdiff / 2)

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(xmax, img.shape[1])
            ymax = min(ymax, img.shape[0])

            mask[ymin:ymax, xmin:xmax] = np.full((ymax - ymin, xmax - xmin), cat)

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

    # Let's compute crop size

    if crop_coordinates is None:

        # We compute the range within which crop size should be

        # Crop size should be > height, width bbox (to keep the bbox within the crop)
        # Crop size should be > crop_dim - delta
        crop_size_min = max(height, width, crop_dim - crop_delta)

        # Crop size should be < crop_dim - delta
        crop_size_max = crop_dim + crop_delta

        # if bbox is bigger than crop size min, we replace crop size min by bbox
        # So, we'll have bbox size <= crop size <= crop size max

        # if bbox is bigger than crop size max, we replace crop size max by bbox
        # So, we'll have bbox size == crop size

        if crop_size_max < max(height, width):
            crop_size_max = max(height, width)
            warnings.warn(
                f"Bbox size ({height}, {width}) > crop dim {crop_size_max} for {img_path}, using crop_dim = bbox size"
            )

        if crop_size_max < crop_size_min:
            raise ValueError(f"Crop size cannot be computed for {img_path}")

        crop_size = random.randint(crop_size_min, crop_size_max)

        if crop_size > min(img.shape[0], img.shape[1]):

            warnings.warn(
                f"Image size ({img.shape}) < crop dim {crop_size} for {img_path}, zero padding is done on image"
            )

            if crop_size > img.shape[0]:
                y_padding = crop_size - img.shape[0]
            else:
                y_padding = 0

            if crop_size > img.shape[1]:
                x_padding = crop_size - img.shape[1]
            else:
                x_padding = 0

            img = np.pad(
                img,
                ((y_padding, x_padding), (y_padding, x_padding), (0, 0)),
                "constant",
                constant_values=0,
            )

            mask = np.pad(
                mask,
                ((y_padding, x_padding), (y_padding, x_padding)),
                "constant",
                constant_values=0,
            )

            x_max_ref += x_padding
            x_min_ref += x_padding
            y_max_ref += y_padding
            y_min_ref += y_padding

        # Let's compute crop position
        # The final crop coordinates will be [x_crop:x_crop+crop_size+margin,y_crop:y_crop+crop_size+margin)

        # The bbox needs to fit in the crop without context
        # So x_crop <= x_min_ref and x_crop + crop_size >= x_max_ref

        # The crop with context needs to fit in the img
        # So x_crop - context_pixels >=0 and x_crop + crop_size + context_pixels <= img_size-1

        x_crop_min = max(context_pixels, x_max_ref - crop_size)
        x_crop_max = min(x_min_ref, img.shape[1] - crop_size - context_pixels)

        y_crop_min = max(context_pixels, y_max_ref - crop_size)
        y_crop_max = min(y_min_ref, img.shape[0] - crop_size - context_pixels)

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
            return x_crop - x_min_ref, y_crop - y_min_ref, crop_size

    else:
        x_crop, y_crop, crop_size = crop_coordinates
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

    mask = Image.fromarray(mask)
    mask = F.resize(mask, output_dim + margin, interpolation=InterpolationMode.NEAREST)

    # resize ref_bbox to output_dim + margin
    ref_bbox = [
        cat_ref,
        int(ref_bbox[0] * (output_dim + margin) / crop_size),
        int(ref_bbox[1] * (output_dim + margin) / crop_size),
        int(ref_bbox[2] * (output_dim + margin) / crop_size),
        int(ref_bbox[3] * (output_dim + margin) / crop_size),
    ]

    return img, mask, ref_bbox


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
            Image.open(path_img)
            if path_bb is not None:
                try:
                    crop_image(
                        path_img,
                        path_bb,
                        mask_random_offset=mask_random_offset,
                        mask_delta=mask_delta,
                        crop_delta=0,
                        mask_square=mask_square,
                        crop_dim=crop_dim + crop_delta,
                        output_dim=output_dim,
                        context_pixels=context_pixels,
                        load_size=load_size,
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
