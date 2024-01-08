import os.path
import warnings

import numpy as np
import torch

# import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import (
    make_labeled_dataset,
    make_labeled_path_dataset,
)
from data.utils import load_image


class SelfSupervisedLabeledClsDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.

    Domain A must have labels, at the moment the subdir of domain A acts as the label string (turned into an int)

    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt, phase)

        if not os.path.isfile(self.dir_A + "/paths.txt"):
            self.A_img_paths, self.A_label = make_labeled_dataset(
                self.dir_A, opt.data_max_dataset_size
            )  # load images from '/path/to/data/trainA' as well as labels
            self.A_label = np.array(self.A_label)

        else:
            self.A_img_paths, self.A_label = make_labeled_path_dataset(
                self.dir_A, "/paths.txt", opt.data_max_dataset_size
            )  # load images from '/path/to/data/trainA/paths.txt' as well as labels
            self.A_label = np.array(self.A_label, dtype=np.float32)

        self.A_size = len(self.A_img_paths)  # get the size of dataset A

        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))

        self.semantic_nclasses = self.opt.cls_semantic_nclasses

    def get_img(
        self,
        A_img_path,
        A_label_mask_path,
        A_label_cls,
        B_img_path,
        B_label_mask_path,
        B_label_cls,
        index,
    ):
        A_img = load_image(A_img_path)
        # apply image transformation
        A = self.transform_A(A_img)
        # get labels
        A_label = self.A_label[index % self.A_size]
        A_label_mask = torch.ones_like(A, dtype=torch.long)
        if A_label > self.semantic_nclasses - 1:
            warnings.warn(
                "A label is above number of semantic classes for img %s" % (A_img_path)
            )
            A_label = self.semantic_nclasses - 1

        return {
            "A": A,
            "A_img_paths": A_img_path,
            "A_label_cls": A_label,
            "A_label_mask": A_label_mask,
            "B": A,
            "B_img_paths": A_img_path,
            "B_label_cls": A_label,
            "B_label_mask": A_label_mask,
        }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size
