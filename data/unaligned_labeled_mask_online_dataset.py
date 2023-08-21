import os.path
import random
import warnings

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image

from data.base_dataset import BaseDataset, get_transform, get_transform_seg
from data.image_folder import make_dataset, make_dataset_path, make_labeled_path_dataset
from data.online_creation import crop_image


class UnalignedLabeledMaskOnlineDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets with mask labels.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.

    Domain A must have labels, at the moment there are two subdirections 'images' and 'labels'.

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

        if os.path.exists(self.dir_A):
            self.A_img_paths, self.A_label_mask_paths = make_labeled_path_dataset(
                self.dir_A, "/paths.txt"
            )  # load images from '/path/to/data/trainA/paths.txt' as well as labels
        else:
            self.A_img_paths, self.A_label_mask_paths = make_labeled_path_dataset(
                opt.dataroot, "/paths.txt"
            )  # load images from '/path/to/data/trainA/paths.txt' as well as labels

        if self.use_domain_B and os.path.exists(self.dir_B):
            self.B_img_paths, self.B_label_mask_paths = make_labeled_path_dataset(
                self.dir_B, "/paths.txt"
            )  # load images from '/path/to/data/trainB'
            if self.B_label_mask_paths == []:
                delattr(self, "B_label_mask_paths")

        if self.opt.data_sanitize_paths:
            self.sanitize()
        elif opt.data_max_dataset_size != float("inf"):
            self.A_img_paths, self.A_label_mask_paths = (
                self.A_img_paths[: opt.data_max_dataset_size],
                self.A_label_mask_paths[: opt.data_max_dataset_size],
            )
            if self.use_domain_B and os.path.exists(self.dir_B):
                self.B_img_paths, self.B_label_mask_paths = (
                    self.B_img_paths[: opt.data_max_dataset_size],
                    self.B_label_mask_paths[: opt.data_max_dataset_size],
                )

        self.A_size = len(self.A_img_paths)  # get the size of dataset A
        if self.use_domain_B and os.path.exists(self.dir_B):
            self.B_size = len(self.B_img_paths)  # get the size of dataset B

        self.transform = get_transform_seg(self.opt, grayscale=(self.input_nc == 1))
        self.transform_noseg = get_transform(self.opt, grayscale=(self.input_nc == 1))

        self.opt = opt

        self.semantic_nclasses = self.opt.f_s_semantic_nclasses

        self.header = ["img", "mask"]

    def get_img(
        self,
        A_img_path,
        A_label_mask_path,
        A_label_cls,
        B_img_path=None,
        B_label_mask_path=None,
        B_label_cls=None,
        index=None,
        clamp_semantics=True,
    ):
        # Domain A
        try:
            if self.opt.data_online_creation_mask_delta_A_ratio == [[]]:
                mask_delta_A = self.opt.data_online_creation_mask_delta_A
            else:
                mask_delta_A = self.opt.data_online_creation_mask_delta_A_ratio

            A_img, A_label_mask, A_ref_bbox = crop_image(
                A_img_path,
                A_label_mask_path,
                mask_delta=mask_delta_A,
                mask_random_offset=self.opt.data_online_creation_mask_random_offset_A,
                crop_delta=self.opt.data_online_creation_crop_delta_A,
                mask_square=self.opt.data_online_creation_mask_square_A,
                crop_dim=self.opt.data_online_creation_crop_size_A,
                output_dim=self.opt.data_load_size,
                context_pixels=self.opt.data_online_context_pixels,
                load_size=self.opt.data_online_creation_load_size_A,
                select_cat=self.opt.data_online_select_category,
                fixed_mask_size=self.opt.data_online_fixed_mask_size,
                inverted_mask=self.opt.data_inverted_mask,
                single_bbox=self.opt.data_online_single_bbox,
            )
            self.cat_A_ref_bbox = torch.tensor(A_ref_bbox[0])
            A_ref_bbox = A_ref_bbox[1:]

        except Exception as e:
            print(e, "domain A data loading for ", A_img_path)
            return None

        A, A_label_mask, A_ref_bbox = self.transform(A_img, A_label_mask, A_ref_bbox)

        if clamp_semantics and torch.any(A_label_mask > self.semantic_nclasses - 1):
            warnings.warn(
                f"A label is above number of semantic classes for img {A_img_path} and label {A_label_mask_path}, label is clamped to have only {self.semantic_nclasses} classes."
            )
            A_label_mask = torch.clamp(A_label_mask, max=self.semantic_nclasses - 1)

        if self.opt.f_s_all_classes_as_one:
            A_label_mask = (A_label_mask >= 1) * 1

        result = {
            "A": A,
            "A_img_paths": A_img_path,
            "A_label_mask": A_label_mask,
            "A_ref_bbox": A_ref_bbox,
        }

        # Domain B
        if B_img_path is not None:
            try:
                if self.opt.data_online_creation_mask_delta_B_ratio == [[]]:
                    mask_delta_B = self.opt.data_online_creation_mask_delta_B
                else:
                    mask_delta_B = self.opt.data_online_creation_mask_delta_B_ratio

                if B_label_mask_path is not None:
                    B_img, B_label_mask, B_ref_bbox = crop_image(
                        B_img_path,
                        B_label_mask_path,
                        mask_delta=mask_delta_B,
                        mask_random_offset=self.opt.data_online_creation_mask_random_offset_B,
                        crop_delta=self.opt.data_online_creation_crop_delta_B,
                        mask_square=self.opt.data_online_creation_mask_square_B,
                        crop_dim=self.opt.data_online_creation_crop_size_B,
                        output_dim=self.opt.data_load_size,
                        context_pixels=self.opt.data_online_context_pixels,
                        load_size=self.opt.data_online_creation_load_size_B,
                        fixed_mask_size=self.opt.data_online_fixed_mask_size,
                        inverted_mask=self.opt.data_inverted_mask,
                        single_bbox=self.opt.data_online_single_bbox,
                    )

                    self.cat_B_ref_bbox = torch.tensor(B_ref_bbox[0])
                    B_ref_bbox = B_ref_bbox[1:]

                    B, B_label_mask, B_ref_bbox = self.transform(
                        B_img, B_label_mask, B_ref_bbox
                    )

                    if clamp_semantics and torch.any(
                        B_label_mask > self.semantic_nclasses - 1
                    ):
                        warnings.warn(
                            f"A label is above number of semantic classes for img {B_img_path} and label {B_label_mask_path}, label is clamped to have only {self.semantic_nclasses} classes."
                        )
                        B_label_mask = torch.clamp(
                            B_label_mask, max=self.semantic_nclasses - 1
                        )

                    if self.opt.f_s_all_classes_as_one:
                        B_label_mask = (B_label_mask >= 1) * 1

                else:
                    B_img = Image.open(B_img_path).convert("RGB")
                    B = self.transform_noseg(B_img)
                    B_label_mask = []

            except Exception as e:
                print(e, "domain B data loading for ", B_img_path)
                return None

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
                        "B_ref_bbox": B_ref_bbox,
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
