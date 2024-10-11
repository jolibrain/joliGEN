import os
import random
import re
import bisect
import torch
from collections import OrderedDict
from data.base_dataset import BaseDataset, get_transform_list
from data.image_folder import make_labeled_path_dataset
from data.online_creation import crop_image
from data.online_creation import fill_mask_with_random, fill_mask_with_color


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split("(\d+)", text)]


class SelfSupervisedVidMaskOnlineDataset(BaseDataset):
    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if hasattr(self, "B_img_paths"):
            return max(self.A_size, self.B_size)
        else:
            return self.A_size

    def __init__(self, opt, phase, name=""):
        BaseDataset.__init__(self, opt, phase, name)

        self.A_img_paths, self.A_label_mask_paths = make_labeled_path_dataset(
            self.dir_A, "/paths.txt"
        )  # load images from '/path/to/data/trainA/paths.txt' as well as labels
        # sort
        self.A_img_paths.sort(key=natural_keys)
        self.A_label_mask_paths.sort(key=natural_keys)

        if self.opt.data_sanitize_paths:
            self.sanitize()
        elif opt.data_max_dataset_size != float("inf"):
            self.A_img_paths, self.A_label_mask_paths = (
                self.A_img_paths[: opt.data_max_dataset_size],
                self.A_label_mask_paths[: opt.data_max_dataset_size],
            )

        self.transform = get_transform_list(self.opt, grayscale=(self.input_nc == 1))

        self.num_frames = opt.data_temporal_number_frames
        self.frame_step = opt.data_temporal_frame_step

        self.num_A = len(self.A_img_paths)
        self.range_A = self.num_A - self.num_frames * self.frame_step

        self.num_common_char = self.opt.data_temporal_num_common_char

        self.opt = opt

        self.A_size = len(self.A_img_paths)  # get the size of dataset A

        # dataset form img(bbox)/vid_series/vid_series_#frame.png(.txt)
        # a ordered list with all video series paths
        self.vid_series_paths = list(
            OrderedDict.fromkeys(os.path.dirname(path) for path in self.A_img_paths)
        )
        # Initialize a dictionary to count how many available paths belong to each directory
        self.frames_counts = {
            vid_serie: (-self.num_frames + 1) * self.frame_step
            for vid_serie in self.vid_series_paths
        }
        # Loop through self.A_img_paths and count the occurrences of each directory
        for path in self.A_img_paths:
            dirname = os.path.dirname(path)
            if dirname in self.vid_series_paths:
                self.frames_counts[dirname] += 1

        # Store cumulative sums of available frames in the order of video series.
        self.cumulative_sums = []
        cumulative_sum = 0
        # Iterate through each video series in frames_counts to compute the cumulative sum of available frame.
        for vid_serie in self.vid_series_paths:
            count_minus = self.frames_counts[vid_serie]
            if count_minus > 0:
                cumulative_sum += count_minus
            self.cumulative_sums.append(cumulative_sum)

        # Iterate through each video series to get available frame pool
        self.available_frame_pool = []
        start_count = 0
        for i in range(len(self.cumulative_sums)):
            num_frames = (
                self.cumulative_sums[i] - self.cumulative_sums[i - 1]
                if i != 0
                else self.cumulative_sums[i]
            )
            end_count = start_count + num_frames
            self.available_frame_pool.append(list(range(start_count, end_count)))
            start_count = end_count

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

        if len(self.frames_counts) == 1:  # single video mario
            if self.range_A == 0:
                index_A = 0
            else:
                index_A = random.randint(0, self.range_A - 1)
        else:  # video series
            range_A = self.cumulative_sums[
                -1
            ]  # total number of frames that can be used as a starting frame
            random_A = random.randint(
                0, range_A - 1
            )  # chose one frame from available video series

            # according to the selected_index, get the video series and frame number
            selected_index = [
                i for i, t in enumerate(self.available_frame_pool) if random_A in t
            ]
            if len(selected_index) == 1:
                selected_index = selected_index[0]
            else:
                raise ValueError("random_A not found in any sublist, check dataset")
            selected_vid = self.vid_series_paths[selected_index]
            if selected_index > 0:
                frame_num = random_A - self.cumulative_sums[selected_index - 1]
            else:
                frame_num = random_A

            filtered_paths = [
                path
                for path in self.A_img_paths
                if os.path.dirname(path) == selected_vid
            ]

            # Get path and index_A
            selected_path = filtered_paths[frame_num]
            index_A = self.A_img_paths.index(selected_path)

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
                cur_A_img, cur_A_label, ref_A_bbox, A_ref_bbox_id = crop_image(
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
                if i == 0:
                    A_ref_bbox = ref_A_bbox[1:]

            except Exception as e:
                print(e, f"{i+1}th frame of domain A in temporal dataloading")
                return None

            images_A.append(cur_A_img)
            labels_A.append(cur_A_label)

        images_A, labels_A, A_ref_bbox = self.transform(images_A, labels_A, A_ref_bbox)
        A_ref_label = labels_A[0]
        A_ref_img = images_A[0]
        images_A = torch.stack(images_A)
        labels_A = torch.stack(labels_A)

        result = {
            "A_ref": A_ref_img,
            "A": images_A,
            "A_img_paths": ref_A_img_path,
            "A_ref_bbox": A_ref_bbox,
            "A_label_mask": labels_A,
            "A_ref_label_mask": A_ref_label,
            "B_ref": A_ref_img,
            "B": images_A,
            "B_img_paths": ref_A_img_path,
            "B_ref_bbox": A_ref_bbox,
            "B_label_mask": labels_A,
            "B_ref_label_mask": A_ref_label,
        }

        try:
            if self.opt.data_online_creation_rand_mask_A:
                A_ref_img = fill_mask_with_random(
                    result["A_ref"], result["A_ref_label_mask"], -1
                )
                images_A = fill_mask_with_random(
                    result["A"], result["A_label_mask"], -1
                )
            elif self.opt.data_online_creation_color_mask_A:
                A_ref_img = fill_mask_with_color(
                    result["A_ref"], result["A_ref_label_mask"], {}
                )
                images_A = fill_mask_with_color(result["A"], result["A_label_mask"], {})
            else:
                raise Exception(
                    "self supervised dataset: no self supervised method specified"
                )

            result.update(
                {
                    "A_ref": A_ref_img,
                    "A": images_A,
                    "A_img_paths": ref_A_img_path,
                }
            )
        except Exception as e:
            print(
                e,
                "self supervised temporal labeled mask online data loading from ",
                ref_A_img_path,
            )
            return None
        return result
