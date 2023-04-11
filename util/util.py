"""This module contains simple helper functions """
from __future__ import print_function

import os
import sys

import numpy as np
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image


def display_mask(mask):
    dict_col = np.array(
        [
            [0, 0, 0],  # black
            [0, 255, 0],  # green
            [255, 0, 0],  # red
            [0, 0, 255],  # blue
            [0, 255, 255],  # cyan
            [255, 255, 255],  # white
            [96, 96, 96],  # grey
            [255, 255, 0],  # yellow
            [237, 127, 16],  # orange
            [102, 0, 153],  # purple
            [88, 41, 0],  # brown
            [253, 108, 158],  # pink
            [128, 0, 0],  # maroon
            [255, 0, 255],
            [255, 0, 127],
            [0, 128, 255],
            [0, 102, 51],  # 17
            [192, 192, 192],
            [128, 128, 0],
            [84, 151, 120],
            [46, 15, 220],
            [135, 171, 56],
            [108, 85, 147],
            [5, 177, 1],
            [138, 202, 147],
            [121, 79, 29],
            [181, 104, 194],
            [145, 208, 44],
            [174, 205, 142],
            [247, 238, 216],
            [147, 222, 253],
            [52, 214, 18],
            [79, 139, 185],
            [242, 197, 204],
            [245, 63, 181],
            [159, 12, 168],
            [86, 211, 74],
            [44, 8, 108],
            [17, 50, 174],
            [109, 11, 174],
            [225, 232, 33],
            [43, 114, 246],
            [251, 129, 86],
            [24, 133, 137],
            [244, 130, 204],
            [44, 66, 142],
            [64, 94, 105],
            [124, 78, 24],
            [54, 113, 215],
            [195, 178, 183],
            [68, 187, 157],
            [58, 2, 74],
            [206, 231, 219],
            [35, 48, 130],
            [77, 82, 179],
            [234, 19, 137],
            [91, 223, 19],
            [223, 181, 222],
            [4, 124, 213],
            [187, 166, 241],
            [93, 92, 72],
            [231, 50, 218],
            [80, 245, 177],
            [73, 55, 73],
            [50, 51, 147],
            [47, 214, 39],
            [247, 199, 36],
            [17, 158, 5],
            [193, 80, 76],
            [170, 248, 247],
            [14, 207, 36],
            [210, 70, 34],
            [238, 181, 102],
            [3, 13, 193],
            [174, 119, 47],
            [149, 128, 252],
            [212, 40, 100],
            [30, 128, 217],
            [136, 86, 198],
            [82, 20, 13],
            [59, 33, 215],
            [155, 248, 53],
            [90, 4, 139],
            [22, 165, 231],
            [222, 26, 214],
            [215, 153, 90],
            [156, 185, 244],
            [106, 193, 18],
            [182, 180, 248],
            [85, 201, 245],
            [237, 85, 73],
            [172, 163, 132],
            [47, 240, 1],
            [156, 142, 125],
            [2, 56, 94],
            [136, 210, 31],
            [84, 171, 87],
            [244, 35, 171],
            [173, 149, 86],
            [233, 252, 246],
            [212, 141, 141],
            [106, 102, 228],
            [121, 83, 155],
            [10, 3, 216],
            [240, 229, 121],
            [47, 1, 184],
            [44, 37, 63],
            [88, 120, 78],
            [182, 63, 128],
            [61, 49, 21],
            [10, 139, 230],
            [121, 148, 133],
            [109, 7, 133],
            [120, 60, 36],
            [67, 0, 182],
            [173, 170, 110],
            [43, 168, 78],
            [221, 107, 14],
            [46, 87, 45],
            [18, 208, 210],
        ]
    )
    nb_cls_display = len(dict_col)

    nb_cls_display = len(dict_col)

    try:
        len(mask.shape) == 2
    except AssertionError:
        print("Mask's shape is not 2")
    mask_dis = np.zeros((mask.shape[0], mask.shape[1], 3))
    # print('mask_dis shape',mask_dis.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[0]):
            cls_display = mask[i, j] % nb_cls_display
            mask_dis[i, j, :] = dict_col[cls_display]
    return mask_dis


def tensor2im(input_image, imtype=np.uint8):
    """ "Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = (
            image_tensor[0].cpu().float().numpy()
        )  # convert it into a numpy array nb : the first image of the batch is displayed
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if len(image_numpy.shape) != 2:  # it is an image
            image_numpy.clip(-1, 1)
            image_numpy = (
                (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
            )  # post-processing: transpose and scaling
        else:  # it is  a mask
            image_numpy = image_numpy.astype(np.uint8)
            image_numpy = display_mask(image_numpy)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def im2tensor(input_image, imtype=torch.float32):
    """
    Convert np.ndarray to tensor
    """
    to_tensor = transforms.ToTensor()
    normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if not isinstance(input_image, torch.Tensor):
        image_tensor = to_tensor(input_image)
        image_tensor = normalizer(image_tensor)
    else:
        image_tensor = input_image
    return image_tensor.to(imtype)


def load_file_from_url(url, directory):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Extract the filename from the URL
    filename = url.split("/")[-1]

    # Download the file
    response = requests.get(url)
    with open(os.path.join(directory, filename), "wb") as f:
        f.write(response.content)


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def gaussian(in_tensor, stddev):
    noisy_image = (
        torch.normal(0, stddev, size=in_tensor.size()).to(in_tensor.device) + in_tensor
    )
    return noisy_image


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


MAX_INT = 1000000000


def flatten_json(src_json, flat_json=None, prefix=""):
    if flat_json is None:
        flat_json = {}
    for key in src_json:
        if isinstance(src_json[key], dict):
            flatten_json(src_json[key], flat_json, prefix + key + "_")
        else:
            flat_json[prefix + key] = src_json[key]
    return flat_json


def delete_flop_param(module):
    if hasattr(module, "total_ops"):
        del module.total_ops

    if hasattr(module, "total_params"):
        del module.total_params

    for child in module.children():
        delete_flop_param(child)


def pairs_of_floats(arg):
    return [float(x) for x in arg.split(",")]


def pairs_of_ints(arg):
    return [int(x) for x in arg.split(",")]
