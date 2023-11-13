import torch
import numpy as np

# Define the CutMix function and its supporting function
# CutMix from https://github.com/boschresearch/unetgan/blob/master/mixup.py


def random_boundingbox(size, lam):
    width, height = size, size
    r = np.sqrt(1.0 - lam)
    w = int(width * r)  # Modified this line
    h = int(height * r)  # And this line
    x = np.random.randint(width)
    y = np.random.randint(height)
    x1 = np.clip(x - w // 2, 0, width)
    y1 = np.clip(y - h // 2, 0, height)
    x2 = np.clip(x + w // 2, 0, width)
    y2 = np.clip(y + h // 2, 0, height)
    return x1, y1, x2, y2


def CutMix(imsize):
    lam = np.random.beta(1, 1)
    x1, y1, x2, y2 = random_boundingbox(imsize, lam)
    lam = 1 - ((x2 - x1) * (y2 - y1) / (imsize * imsize))
    mask = torch.ones((imsize, imsize))
    mask[x1:x2, y1:y2] = 0
    if torch.rand(1) > 0.5:
        mask = 1 - mask
    return mask
