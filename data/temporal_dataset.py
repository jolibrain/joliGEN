from data.base_dataset import BaseDataset
import random
import torch
import re
import os
from data.image_folder import make_labeled_path_dataset
from data.base_dataset import get_transform_list
from data.online_creation import crop_image


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split("(\d+)", text)]


class TemporalDataset(BaseDataset):
    def __len__(self):
        return 1000000000000

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        super(TemporalDataset).__init__()
        self.opt = opt
        btoA = self.opt.data_direction == "BtoA"
        self.input_nc = (
            self.opt.model_output_nc if btoA else self.opt.model_input_nc
        )  # get the number of channels of input image
        output_nc = (
            self.opt.model_input_nc if btoA else self.opt.model_output_nc
        )  # get the number of channels of output image

        self.dir_A = os.path.join(
            opt.dataroot, opt.phase + "A"
        )  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(
            opt.dataroot, opt.phase + "B"
        )  # create a path '/path/to/data/trainB'

        self.A_img_paths, self.A_label_paths = make_labeled_path_dataset(
            self.dir_A, "/paths.txt"
        )  # load images from '/path/to/data/trainA/paths.txt' as well as labels

        self.B_img_paths, self.B_label_paths = make_labeled_path_dataset(
            self.dir_B, "/paths.txt"
        )  # load images from '/path/to/data/trainB'

        self.transform = get_transform_list(self.opt, grayscale=(self.input_nc == 1))

        self.num_A = len(self.A_img_paths)
        self.num_B = len(self.B_img_paths)
        self.num_frames = opt.D_temporal_number_frames
        self.frame_step = opt.D_temporal_frame_step
        self.range_A = self.num_A - self.num_frames * self.frame_step
        self.range_B = self.num_B - self.num_frames * self.frame_step
        self.num_common_char = self.opt.D_temporal_num_common_char

        self.opt = opt

        self.A_size = (
            100  # use to compute image path in base datset method (unused then)
        )
        self.B_size = 100

        # sort
        self.A_img_paths.sort(key=natural_keys)
        self.A_label_paths.sort(key=natural_keys)
        self.B_img_paths.sort(key=natural_keys)
        self.B_label_paths.sort(key=natural_keys)

    def get_img(
        self, A_img_path, A_label_path, B_img_path=None, B_label_path=None, index=None
    ):  # all params are unused

        index_A = random.randint(0, self.range_A - 1)

        images_A = []
        labels_A = []

        ref_name_A = self.A_img_paths[index_A].split("/")[-1][: self.num_common_char]

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
                self.A_label_paths[cur_index_A],
            )

            if self.opt.data_relative_paths:
                cur_A_img_path = os.path.join(self.root, cur_A_img_path)
                if cur_A_label_path is not None:
                    cur_A_label_path = os.path.join(self.root, cur_A_label_path)

            try:
                if i == 0:
                    crop_coordinates = crop_image(
                        cur_A_img_path,
                        cur_A_label_path,
                        mask_delta=self.opt.data_online_creation_mask_delta_A,
                        crop_delta=self.opt.data_online_creation_crop_delta_A,
                        mask_square=self.opt.data_online_creation_mask_square_A,
                        crop_dim=self.opt.data_online_creation_crop_size_A,
                        output_dim=self.opt.data_load_size,
                        context_pixels=self.opt.data_online_context_pixels,
                        get_crop_coordinates=True,
                    )
                cur_A_img, cur_A_label = crop_image(
                    cur_A_img_path,
                    cur_A_label_path,
                    mask_delta=self.opt.data_online_creation_mask_delta_A,
                    crop_delta=self.opt.data_online_creation_crop_delta_A,
                    mask_square=self.opt.data_online_creation_mask_square_A,
                    crop_dim=self.opt.data_online_creation_crop_size_A,
                    output_dim=self.opt.data_load_size,
                    context_pixels=self.opt.data_online_context_pixels,
                    crop_coordinates=crop_coordinates,
                )

            except Exception as e:
                print(e, f"{i+1}th frame of domain A in temporal dataloading")
                return None

            images_A.append(cur_A_img)
            labels_A.append(cur_A_label)

        images_A, _ = self.transform(images_A, labels_A)

        images_A = torch.stack(images_A)

        index_B = random.randint(0, self.range_B - 1)

        images_B = []
        labels_B = []

        ref_name_B = self.B_img_paths[index_B].split("/")[-1][: self.num_common_char]

        for i in range(self.num_frames):
            cur_index_B = index_B + i * self.frame_step

            if (
                self.num_common_char != -1
                and self.B_img_paths[cur_index_B].split("/")[-1][: self.num_common_char]
                not in ref_name_B
            ):
                return None

            cur_B_img_path, cur_B_label_path = (
                self.B_img_paths[cur_index_B],
                self.B_label_paths[cur_index_B],
            )

            if self.opt.data_relative_paths:
                cur_B_img_path = os.path.join(self.root, cur_B_img_path)
                if cur_B_label_path is not None:
                    cur_B_label_path = os.path.join(self.root, cur_B_label_path)

            try:
                if i == 0:
                    crop_coordinates = crop_image(
                        cur_B_img_path,
                        cur_B_label_path,
                        mask_delta=self.opt.data_online_creation_mask_delta_B,
                        crop_delta=self.opt.data_online_creation_crop_delta_B,
                        mask_square=self.opt.data_online_creation_mask_square_B,
                        crop_dim=self.opt.data_online_creation_crop_size_B,
                        output_dim=self.opt.data_load_size,
                        context_pixels=self.opt.data_online_context_pixels,
                        get_crop_coordinates=True,
                    )

                cur_B_img, cur_B_label = crop_image(
                    cur_B_img_path,
                    cur_B_label_path,
                    mask_delta=self.opt.data_online_creation_mask_delta_B,
                    crop_delta=self.opt.data_online_creation_crop_delta_B,
                    mask_square=self.opt.data_online_creation_mask_square_B,
                    crop_dim=self.opt.data_online_creation_crop_size_B,
                    output_dim=self.opt.data_load_size,
                    context_pixels=self.opt.data_online_context_pixels,
                    crop_coordinates=crop_coordinates,
                )

            except Exception as e:
                print(e, f"{i+1}th frame of domain B in temporal dataloading")
                return None

            images_B.append(cur_B_img)
            labels_B.append(cur_B_label)

        images_B, _ = self.transform(images_B, labels_B)

        images_B = torch.stack(images_B)

        return {"A": images_A, "B": images_B}
