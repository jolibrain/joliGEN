import os.path

# import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import (
    make_dataset,
    make_labeled_dataset,
    make_labeled_path_dataset,
)
from PIL import Image
import random
import numpy as np
import warnings


class UnalignedLabeledDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.

    Domain A must have labels, at the moment the subdir of domain A acts as the label string (turned into an int)

    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

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

        # print('A_label',self.A_label)
        if opt.train_sem_use_label_B:
            if not os.path.isfile(self.dir_B + "/paths.txt"):
                self.B_img_paths, self.B_label = make_labeled_dataset(
                    self.dir_B, opt.data_max_dataset_size
                )
                self.B_label = np.array(self.B_label)
            else:
                self.B_img_paths, self.B_label = make_labeled_path_dataset(
                    self.dir_B, "/paths.txt", opt.data_max_dataset_size
                )  # load images from '/path/to/data/trainB'
                self.B_label = np.array(self.B_label, dtype=np.float32)

        else:
            self.B_img_paths = sorted(
                make_dataset(self.dir_B, opt.data_max_dataset_size)
            )  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_img_paths)  # get the size of dataset A
        self.B_size = len(self.B_img_paths)  # get the size of dataset B

        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1))

        self.semantic_nclasses = self.opt.f_s_semantic_nclasses

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
        A_img = Image.open(A_img_path).convert("RGB")
        B_img = Image.open(B_img_path).convert("RGB")
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # get labels
        A_label = self.A_label[index % self.A_size]
        if A_label > self.semantic_nclasses - 1:
            warnings.warn(
                "A label is above number of semantic classes for img %s" % (A_img_path)
            )
            A_label = self.semantic_nclasses - 1

        if hasattr(self, "B_label"):
            B_label = self.B_label[index % self.B_size]
            if B_label > self.semantic_nclasses - 1:
                warnings.warn(
                    "A label is above number of semantic classes for img %s"
                    % (B_img_path)
                )
                B_label = self.semantic_nclasses - 1

            return {
                "A": A,
                "B": B,
                "A_img_paths": A_img_path,
                "B_paths": B_img_path,
                "A_label_cls": A_label,
                "B_label_cls": B_label,
            }

        return {
            "A": A,
            "B": B,
            "A_img_paths": A_img_path,
            "B_paths": B_img_path,
            "A_label_cls": A_label,
        }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
