import os.path
from data.base_dataset import BaseDataset, get_transform
from data.utils import load_image
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, phase, name=""):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt, phase, name)

        self.A_img_paths = sorted(
            make_dataset(self.dir_A, opt.data_max_dataset_size)
        )  # load images from '/path/to/data/trainA'
        self.B_img_paths = sorted(
            make_dataset(self.dir_B, opt.data_max_dataset_size)
        )  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_img_paths)  # get the size of dataset A
        self.B_size = len(self.B_img_paths)  # get the size of dataset B

        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1))

        self.header = ["img"]

        if self.opt.data_relative_paths:
            self.B_img_prompt = {
                f"{self.root}{key}": value for key, value in self.B_img_prompt.items()
            }

    # A_label_path and B_label_path are unused
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
        B_img = load_image(B_img_path)
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {"A": A, "B": B, "A_img_paths": A_img_path, "B_img_paths": B_img_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
