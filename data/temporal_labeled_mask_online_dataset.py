import os
import random
import re

import torch

from data.base_dataset import BaseDataset, get_transform_list
from data.image_folder import make_labeled_path_dataset
from data.online_creation import crop_image


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split("(\d+)", text)]


class TemporalLabeledMaskOnlineDataset(BaseDataset):
    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if hasattr(self, "B_img_paths"):
            return max(self.A_size, self.B_size)
        else:
            return self.A_size

    def __init__(self, opt, phase):
        BaseDataset.__init__(self, opt, phase)

        self.A_img_paths, self.A_label_mask_paths = make_labeled_path_dataset(
            self.dir_A, "/paths.txt"
        )  # load images from '/path/to/data/trainA/paths.txt' as well as labels

        if self.use_domain_B:
            self.B_img_paths, self.B_label_mask_paths = make_labeled_path_dataset(
                self.dir_B, "/paths.txt"
            )  # load images from '/path/to/data/trainB'

        # sort
        self.A_img_paths.sort(key=natural_keys)
        self.A_label_mask_paths.sort(key=natural_keys)

        if self.use_domain_B:
            self.B_img_paths.sort(key=natural_keys)
            self.B_label_mask_paths.sort(key=natural_keys)

        self.A_img_paths, self.A_label_mask_paths = (
            self.A_img_paths[: opt.data_max_dataset_size],
            self.A_label_mask_paths[: opt.data_max_dataset_size],
        )

        if self.use_domain_B:
            self.B_img_paths, self.B_label_mask_paths = (
                self.B_img_paths[: opt.data_max_dataset_size],
                self.B_label_mask_paths[: opt.data_max_dataset_size],
            )

        self.transform = get_transform_list(self.opt, grayscale=(self.input_nc == 1))

        self.num_frames = opt.data_temporal_number_frames
        self.frame_step = opt.data_temporal_frame_step

        self.num_A = len(self.A_img_paths)
        self.range_A = self.num_A - self.num_frames * self.frame_step

        if self.use_domain_B:
            self.num_B = len(self.B_img_paths)
            self.range_B = self.num_B - self.num_frames * self.frame_step
        self.num_common_char = self.opt.data_temporal_num_common_char

        self.opt = opt

        self.A_size = len(self.A_img_paths)  # get the size of dataset A
        if self.use_domain_B and os.path.exists(self.dir_B):
            self.B_size = len(self.B_img_paths)  # get the size of dataset B

    def get_img(
        self,
        A_img_path,
        A_label_mask_path,
        A_label_cls,
        B_img_path=None,
        B_label_mask_path=None,
        B_label_cls=None,
        index=None,
    ):  # all params are unused
        index_A = random.randint(0, self.range_A - 1)

        images_A = []
        labels_A = []

        ref_A_img_path = self.A_img_paths[index_A]

        ref_name_A = ref_A_img_path.split("/")[-1][: self.num_common_char]

        for i in range(self.num_frames):
            cur_index_A = index_A + i * self.frame_step

            if (
                self.num_common_char != -1
                and self.A_img_paths[cur_index_A].split("/")[-1][: self.num_common_char]
                not in ref_name_A
            ):
                return None

            cur_A_img_path, cur_A_label_path = (
                self.A_img_paths[cur_index_A],
                self.A_label_mask_paths[cur_index_A],
            )

            if self.opt.data_relative_paths:
                cur_A_img_path = os.path.join(self.root, cur_A_img_path)
                if cur_A_label_path is not None:
                    cur_A_label_path = os.path.join(self.root, cur_A_label_path)

            try:
                if self.opt.data_online_creation_mask_delta_A_ratio == [[]]:
                    mask_delta_A = self.opt.data_online_creation_mask_delta_A
                else:
                    mask_delta_A = self.opt.data_online_creation_mask_delta_A_ratio
                if i == 0:
                    crop_coordinates = crop_image(
                        cur_A_img_path,
                        cur_A_label_path,
                        mask_delta=mask_delta_A,
                        mask_random_offset=self.opt.data_online_creation_mask_random_offset_A,
                        crop_delta=self.opt.data_online_creation_crop_delta_A,
                        mask_square=self.opt.data_online_creation_mask_square_A,
                        crop_dim=self.opt.data_online_creation_crop_size_A,
                        output_dim=self.opt.data_load_size,
                        context_pixels=self.opt.data_online_context_pixels,
                        load_size=self.opt.data_online_creation_load_size_A,
                        get_crop_coordinates=True,
                        fixed_mask_size=self.opt.data_online_fixed_mask_size,
                    )
                cur_A_img, cur_A_label, ref_A_bbox = crop_image(
                    cur_A_img_path,
                    cur_A_label_path,
                    mask_delta=mask_delta_A,
                    mask_random_offset=self.opt.data_online_creation_mask_random_offset_A,
                    crop_delta=self.opt.data_online_creation_crop_delta_A,
                    mask_square=self.opt.data_online_creation_mask_square_A,
                    crop_dim=self.opt.data_online_creation_crop_size_A,
                    output_dim=self.opt.data_load_size,
                    context_pixels=self.opt.data_online_context_pixels,
                    load_size=self.opt.data_online_creation_load_size_A,
                    crop_coordinates=crop_coordinates,
                    fixed_mask_size=self.opt.data_online_fixed_mask_size,
                )

            except Exception as e:
                print(e, f"{i+1}th frame of domain A in temporal dataloading")
                return None

            images_A.append(cur_A_img)
            labels_A.append(cur_A_label)

        images_A, labels_A = self.transform(images_A, labels_A)

        images_A = torch.stack(images_A)

        labels_A = torch.stack(labels_A)

        if self.use_domain_B:
            index_B = random.randint(0, self.range_B - 1)

            images_B = []
            labels_B = []

            ref_B_img_path = self.B_img_paths[index_B]

            ref_name_B = ref_B_img_path.split("/")[-1][: self.num_common_char]

            for i in range(self.num_frames):
                cur_index_B = index_B + i * self.frame_step

                if (
                    self.num_common_char != -1
                    and self.B_img_paths[cur_index_B].split("/")[-1][
                        : self.num_common_char
                    ]
                    not in ref_name_B
                ):
                    return None

                cur_B_img_path, cur_B_label_path = (
                    self.B_img_paths[cur_index_B],
                    self.B_label_mask_paths[cur_index_B],
                )

                if self.opt.data_relative_paths:
                    cur_B_img_path = os.path.join(self.root, cur_B_img_path)
                    if cur_B_label_path is not None:
                        cur_B_label_path = os.path.join(self.root, cur_B_label_path)

                try:
                    if self.opt.data_online_creation_mask_delta_B_ratio == [[]]:
                        mask_delta_B = self.opt.data_online_creation_mask_delta_B
                    else:
                        mask_delta_B = self.opt.data_online_creation_mask_delta_B_ratio
                    if i == 0:
                        crop_coordinates = crop_image(
                            cur_B_img_path,
                            cur_B_label_path,
                            mask_delta=mask_delta_B,
                            mask_random_offset=self.opt.data_online_creation_mask_random_offset_B,
                            crop_delta=self.opt.data_online_creation_crop_delta_B,
                            mask_square=self.opt.data_online_creation_mask_square_B,
                            crop_dim=self.opt.data_online_creation_crop_size_B,
                            output_dim=self.opt.data_load_size,
                            context_pixels=self.opt.data_online_context_pixels,
                            load_size=self.opt.data_online_creation_load_size_B,
                            get_crop_coordinates=True,
                        )

                    cur_B_img, cur_B_label, ref_B_bbox = crop_image(
                        cur_B_img_path,
                        cur_B_label_path,
                        mask_delta=mask_delta_B,
                        mask_random_offset=self.opt.data_online_creation_mask_random_offset_B,
                        crop_delta=self.opt.data_online_creation_crop_delta_B,
                        mask_square=self.opt.data_online_creation_mask_square_B,
                        crop_dim=self.opt.data_online_creation_crop_size_B,
                        output_dim=self.opt.data_load_size,
                        context_pixels=self.opt.data_online_context_pixels,
                        load_size=self.opt.data_online_creation_load_size_B,
                        crop_coordinates=crop_coordinates,
                        fixed_mask_size=self.opt.data_online_fixed_mask_size,
                    )

                except Exception as e:
                    print(e, f"{i+1}th frame of domain B in temporal dataloading")
                    return None

                images_B.append(cur_B_img)
                labels_B.append(cur_B_label)

            images_B, labels_B = self.transform(images_B, labels_B)

            images_B = torch.stack(images_B)

            labels_B = torch.stack(labels_B)

        else:
            images_B = None
            labels_B = None
            ref_B_img_path = None

        result = {
            "A": images_A,
            "A_img_paths": ref_A_img_path,
            "B": images_B,
            "B_img_paths": ref_B_img_path,
        }

        result.update(
            {
                "A_label_mask": labels_A,
                "B_label_mask": labels_B,
            }
        )

        return result
