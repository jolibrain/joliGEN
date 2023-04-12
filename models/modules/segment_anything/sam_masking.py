import os
import subprocess
import sys

import numpy as np
import torch

# get the path of the current file
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, current_path + "/../")
# sys.path.insert(0, current_path)
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor

from util.util import downloadFileWithProgress


def show_mask(mask, cat):
    # convert true/false mask to cat/0 array
    cat_mask = np.zeros_like(mask)
    cat_mask[mask] = cat
    return cat_mask.astype(np.uint8)
    """
    color = np.array([255, 255, 255])  # white
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image
    """


def tensor2cv(img):
    """
    Convert tensor image to cv2 image
    """
    img = img.cpu().detach().numpy()
    img = img.transpose(1, 2, 0)
    img = img * 255
    img = img.astype(np.uint8)
    return img


def cv2tensor(img):
    """
    Convert cv2 image to tensor image
    """
    img = img.astype(np.float32)
    img = img / 255
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    return img


def download_sam_weights():
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    if not os.path.exists(sam_checkpoint):
        url = "https://dl.fbaipublicfiles.com/segment_anything/" + sam_checkpoint
        downloadFileWithProgress(url, "./")
    else:
        print("Using cache found in", sam_checkpoint)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    predictor = SamPredictor(sam)

    return predictor


def gen_mask_from_bbox(img, bbox, predictor, cat=1):
    """
    Generate mask from bounding box
    :param img: image tensor(Size[3, H, W])
    :param bbox: bounding box np.array([x1, y1, x2, y2])
    :return: mask
    """

    if img.shape[0] <= 4:
        cv_img = tensor2cv(img)
    else:
        cv_img = img
    predictor.set_image(cv_img)

    with torch.no_grad():
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,
        )  # outputs a boolean mask (True/False)
    mask = show_mask(masks[0], cat)

    return mask


if __name__ == "__main__":
    import cv2
    from PIL import Image

    img = Image.open("image.jpg")
    img = np.array(img)
    img = cv2tensor(img)
    img = img[None, :, :, :]

    bbox = np.array([1170, 2233, 1452, 2376])

    mask = gen_mask_from_bbox(img.squeeze(0), bbox, None)

    # show the mask on image and save it
    img = tensor2cv(img.squeeze(0))
    mask = tensor2cv(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    cv2.imwrite("mask.jpg", mask)
