import math
import numpy as np
import random
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


def crop_image(
    img_path, bbox_path, mask_delta, crop_delta, mask_square, crop_dim, output_dim
):

    img = np.array(Image.open(img_path))

    with open(bbox_path, "r") as f:
        bboxes = []

        for line in f:
            if len(line) > 2:  # to make sure the current line is a real bbox
                bboxes.append(line)
            elif line != "" or line != " ":
                print("%s does not describe a bbox" % line)

        if len(bboxes) == 0:
            print("There is no bbox.")

        # Creation of a blank mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # A bbox of reference will be used to compute the crop
        idx_bbox_ref = random.randint(0, len(bboxes) - 1)

        for i, cur_bbox in enumerate(bboxes):
            bbox = cur_bbox.split()
            cat = int(bbox[0])
            xmin = math.floor(int(bbox[1]))
            ymin = math.floor(int(bbox[2]))
            xmax = math.floor(int(bbox[3]))
            ymax = math.floor(int(bbox[4]))

            if i == idx_bbox_ref:
                x_min_ref = xmin
                x_max_ref = xmax
                y_min_ref = ymin
                y_max_ref = ymax

            if (
                mask_delta > 0
            ):  # increase mask box so that it can fit the reconstructed object (for semantic loss)
                ymin -= mask_delta
                ymax += mask_delta
                xmin -= mask_delta
                xmax += mask_delta

            if mask_square:
                sdiff = (xmax - xmin) - (ymax - ymin)
                if sdiff > 0:
                    ymax += int(sdiff / 2)
                    ymin -= int(sdiff / 2)
                else:
                    xmax += -int(sdiff / 2)
                    xmin -= -int(sdiff / 2)

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(xmax, img.shape[1])
            ymax = min(ymax, img.shape[0])

            mask[ymin:ymax, xmin:xmax] = np.full((ymax - ymin, xmax - xmin), cat)

    height = y_max_ref - y_min_ref
    width = x_max_ref - x_min_ref

    crop_size_min = max(height, width, crop_dim - crop_delta)
    crop_size_max = max(height, width, crop_dim + crop_delta)

    crop_size = random.randint(crop_size_min, crop_size_max)

    x_crop_min = max(0, x_max_ref - crop_size)
    x_crop_max = min(x_min_ref, img.shape[1] - crop_size)

    y_crop_min = max(0, y_max_ref - crop_size)
    y_crop_max = min(y_min_ref, img.shape[0] - crop_size)

    x_crop = random.randint(x_crop_min, x_crop_max)
    y_crop = random.randint(y_crop_min, y_crop_max)

    img = img[y_crop : y_crop + crop_size, x_crop : x_crop + crop_size, :]
    img = Image.fromarray(img)
    img = F.resize(img, output_dim)

    mask = mask[y_crop : y_crop + crop_size, x_crop : x_crop + crop_size]
    mask = Image.fromarray(mask)
    mask = F.resize(mask, output_dim, interpolation=InterpolationMode.NEAREST)

    return img, mask


def sanitize_paths(
    paths_img,
    paths_bb=None,
    mask_delta=None,
    crop_delta=None,
    mask_square=None,
    crop_dim=None,
    output_dim=None,
    max_dataset_size=float("inf"),
    verbose=False,
):
    return_paths_img = []
    return_paths_bb = []

    if paths_bb is None:
        paths_bb = [None for k in range(len(paths_img))]

    for path_img, path_bb in zip(paths_img, paths_bb):
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
                        mask_delta=mask_delta,
                        crop_delta=0,
                        mask_square=mask_square,
                        crop_dim=crop_dim + crop_delta,
                        output_dim=output_dim,
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
