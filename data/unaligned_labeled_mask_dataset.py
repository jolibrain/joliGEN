import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_seg
from data.image_folder import make_dataset, make_labeled_path_dataset, make_dataset_path
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as F
import warnings


class UnalignedLabeledMaskDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets with mask labels.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.

    Domain A must have labels, at the moment there are two subdirections 'images' and 'labels'.

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

        if os.path.exists(self.dir_A):
            self.A_img_paths, self.A_label = make_labeled_path_dataset(
                self.dir_A, "/paths.txt", opt.data_max_dataset_size
            )  # load images from '/path/to/data/trainA/paths.txt' as well as labels
        else:
            self.A_img_paths, self.A_label = make_labeled_path_dataset(
                opt.dataroot, "/paths.txt", opt.data_max_dataset_size
            )  # load images from '/path/to/data/trainA/paths.txt' as well as labels
        self.A_size = len(self.A_img_paths)  # get the size of dataset A

        if self.use_domain_B and os.path.exists(self.dir_B):
            self.B_img_paths, self.B_label = make_labeled_path_dataset(
                self.dir_B, "/paths.txt", opt.data_max_dataset_size
            )  # load images from '/path/to/data/trainB'
            if self.B_label == []:
                delattr(self, "B_label")
            self.B_size = len(self.B_img_paths)  # get the size of dataset B

        self.A_label_mask_paths = []
        self.B_label_mask_paths = []

        for label in self.A_label:
            self.A_label_mask_paths.append(label.split(" ")[-1])

        if self.use_domain_B and hasattr(self, "B_label"):
            for label in self.B_label:
                self.B_label_mask_paths.append(label.split(" ")[-1])

        self.transform = get_transform_seg(self.opt, grayscale=(self.input_nc == 1))
        self.transform_noseg = get_transform(self.opt, grayscale=(self.input_nc == 1))

        self.semantic_nclasses = self.opt.f_s_semantic_nclasses

    def get_img(
        self,
        A_img_path,
        A_label_mask_path,
        A_label_cls,
        B_img_path=None,
        B_label_mask_path=None,
        B_label_cls=None,
        index=None,
    ):

        # Domain A
        try:
            A_img = Image.open(A_img_path).convert("RGB")
            A_label_mask = Image.open(A_label_mask_path)
        except Exception as e:
            print(
                "failure with reading A domain image ",
                A_img_path,
                " or label ",
                A_label_mask_path,
            )
            print(e)
            return None

        A, A_label_mask = self.transform(A_img, A_label_mask)

        if torch.any(A_label_mask > self.semantic_nclasses - 1):
            warnings.warn(
                "A label is above number of semantic classes for img %s and label %s"
                % (A_img_path, A_label_mask_path)
            )
            A_label_mask = torch.clamp(A_label_mask, max=self.semantic_nclasses - 1)

        if self.opt.f_s_all_classes_as_one:
            A_label_mask = (A_label_mask >= 1) * 1

        result = {"A": A, "A_img_paths": A_img_path, "A_label_mask": A_label_mask}

        # Domain B
        if B_img_path is not None:
            try:
                B_img = Image.open(B_img_path).convert("RGB")
            except:
                print(
                    "failed to read B domain image ",
                    B_img_path,
                )
                return None

            if B_label_mask_path is not None:
                try:
                    B_label_mask = Image.open(B_label_mask_path)
                except:
                    print(
                        f"failed to read domain B label %s for image %s"
                        % (B_label_mask_path, N_img_path)
                    )

                B, B_label_mask = self.transform(B_img, B_label_mask)
                if torch.any(B_label_mask > self.semantic_nclasses - 1):
                    warnings.warn(
                        f"A label is above number of semantic classes for img {B_img_path} and label {B_label_mask_path}, label is clamped to have only {self.semantic_nclasses} classes."
                    )
                    B_label_mask = torch.clamp(
                        B_label_mask, max=self.semantic_nclasses - 1
                    )
            else:
                B = self.transform_noseg(B_img)
                B_label_mask = []

            result.update(
                {
                    "B": B,
                    "B_img_paths": B_img_path,
                }
            )
            if B_label_mask_path is not None:
                result.update(
                    {
                        "B_label_mask": B_label_mask,
                    }
                )

        return result

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if hasattr(self, "B_img_paths"):
            return max(self.A_size, self.B_size)
        else:
            return self.A_size
