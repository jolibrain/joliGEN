import os
import urllib.request

import cv2
import numpy as np
import torch
from torchvision.transforms import Grayscale


def fill_img_with_sketch(img, mask):
    grayscale = Grayscale(3)
    gray = grayscale(img)

    threshold = torch.tensor((120 / 255) * 2 - 1)

    thresh = (gray < threshold) * 1.0  # thresh = ((gray < threshold) * 1.0) * 2 - 1

    mask = torch.clamp(mask, 0, 1)

    return mask * thresh + (1 - mask) * img


def fill_img_with_edges(img, mask):
    device = img.device

    edges_list = []

    for cur_img in img:
        cur_img = (
            (torch.einsum("chw->hwc", cur_img).cpu().numpy() + 1) * 255 / 2
        ).astype(np.uint8)
        edges = cv2.Canny(cur_img, 100, 150)
        edges = (
            (((torch.tensor(edges, device=device) / 255) * 2) - 1)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        edges_list.append(edges)

    edges = torch.cat(edges_list, dim=0)

    mask = torch.clamp(mask, 0, 1)

    return mask * edges + (1 - mask) * img


def fill_img_with_canny(img, mask):
    img_orig = img.clone()
    mask_2D = mask.cpu()[0, :, :][0]  # Convert mask to a 2D array
    coords = np.column_stack(
        np.where(mask_2D > 0)
    )  # Get the coordinates of the white pixels in the mask
    x_0, y_0, w, h = cv2.boundingRect(coords.astype(np.int))
    ## TODO check if [:, :, w, h] or invert w and h ?
    to_sketch = img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h]

    to_sketch = np.transpose(to_sketch.squeeze().cpu().numpy(), (1, 2, 0))
    edges = cv2.Canny((to_sketch * 255).astype(np.uint8), 250, 500)
    # edges = np.transpose(edges, (2, 0, 1))
    edges = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0)
    img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h] = edges

    return img_orig


def fill_img_with_hed(img, mask):
    """
    From pretrained Caffe model with openCV.
    """

    img_orig = img.clone()
    mask_2D = mask.cpu()[0, :, :][0]  # Convert mask to a 2D array
    coords = np.column_stack(
        np.where(mask_2D > 0)
    )  # Get the coordinates of the white pixels in the mask
    x_0, y_0, w, h = cv2.boundingRect(coords.astype(np.int))
    to_sketch = img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h]

    to_sketch = np.transpose(to_sketch.squeeze().cpu().numpy(), (1, 2, 0))

    img_to_sketch = cv2.cvtColor(to_sketch, cv2.COLOR_RGB2BGR)

    (H, W) = img_to_sketch.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img_to_sketch, scalefactor=1.0, size=(W, H), swapRB=False, crop=False
    )

    proto_url = "https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt"
    weights_url = "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel"

    p_filename = "deploy.protoxt"
    w_filename = "pretrained_hed.caffemodel"
    folder = "../models/modules/caffe"

    p_path = os.path.join(folder, p_filename)
    w_path = os.path.join(folder, w_filename)

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Download file from URL and save it as local file
    if not os.path.exists(p_path):
        urllib.request.urlretrieve(proto_url, p_path)
    if not os.path.exists(w_path):
        urllib.request.urlretrieve(weights_url, w_path)
    ## Load the pretrained Caffe model
    net = cv2.dnn.readNetFromCaffe(p_path, w_path)
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")

    hed = torch.from_numpy(hed).unsqueeze(0).unsqueeze(0)

    img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h] = hed

    return img_orig
