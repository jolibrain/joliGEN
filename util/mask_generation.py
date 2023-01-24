import torch
from torchvision.transforms import Grayscale
import cv2
import numpy as np


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
