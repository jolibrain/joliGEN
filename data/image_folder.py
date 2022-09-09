"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
import glob
import re

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    if max_dataset_size == "inf":
        max_dataset_size = len(images)
    return images[: min(max_dataset_size, len(images))]


def make_labeled_dataset(dir, max_dataset_size=float("inf")):
    images = []
    labels = []
    alllabels = {}
    lbl = 0
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    # for root, _, fnames in sorted(os.walk(dir)):
    #    for fname in fnames:
    #        if is_image_file(fname):
    #            path = os.path.join(root, fname)
    #            images.append(path)
    all_files = glob.glob(dir + "/*/*.*")
    for img in all_files:
        if is_image_file(img):
            images.append(img)
            label = os.path.basename(os.path.dirname(img))
            if not label in alllabels:
                alllabels[label] = lbl
                lbl += 1
            label = alllabels[label]
            labels.append(label)

    # print('labels=',labels)
    return (
        images[: min(max_dataset_size, len(images))],
        labels[: min(max_dataset_size, len(images))],
    )


def make_labeled_path_dataset(dir, paths, max_dataset_size=float("inf")):
    images = []
    labels = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    with open(dir + paths, "r") as f:
        paths_list = f.read().split("\n")

    for line in paths_list:
        line_split = line.split(" ")

        if len(line_split) == 2:
            images.append(line_split[0])
            labels.append(line_split[1])
        if len(line_split) == 3:
            images.append(line_split[0])
            labels.append(line_split[1] + " " + line_split[2])

        elif (
            len(line_split) == 1 and len(line_split[0]) > 0
        ):  # we allow B not having a label
            images.append(line_split[0])

    return (
        images[: min(max_dataset_size, len(images))],
        labels[: min(max_dataset_size, len(images))],
    )


def make_dataset_path(dir, paths, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    with open(dir + paths, "r") as f:
        paths_list = f.read().split("\n")

    for line in paths_list:
        if is_image_file(line):
            images.append(line)

    if max_dataset_size == "inf":
        max_dataset_size = len(images)
    return images[: min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)
                )
            )

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split("([0-9]+)", s)]


def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    l.sort(key=alphanum_key)
