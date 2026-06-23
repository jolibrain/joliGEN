import os
import random
import re
import torch
from data.base_dataset import (
    BaseDataset,
    build_masked_global_context_image,
    get_transform_list,
    transform_global_context_images,
)
from data.image_folder import make_labeled_path_dataset
from data.online_creation import crop_image
from data.online_creation import fill_mask_with_random, fill_mask_with_color
from data.temporal_sampling import (
    TemporalFrameStepMixin,
    build_temporal_series_index,
    select_temporal_start_from_series,
)
from util.b2b_context import b2b_global_context_enabled_from_opt


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split("(\d+)", text)]


class SelfSupervisedVidLabeledMaskClsOnlineDataset(TemporalFrameStepMixin, BaseDataset):
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
        self.header = ["img", "cls", "mask"]

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

        self._init_temporal_frame_step_sampling(opt)

        self.num_A = len(self.A_img_paths)
        self.range_A = self.num_A - self.num_frames * self.frame_step

        self.num_common_char = self.opt.data_temporal_num_common_char

        self.opt = opt

        self.A_size = len(self.A_img_paths)  # get the size of dataset A

        # dataset form img(bbox)/vid_series/vid_series_#frame.png(.txt)
        # a ordered list with all video series paths
        (
            self.vid_series_paths,
            self.frames_counts,
            self.cumulative_sums,
            self.available_frame_pool,
        ) = build_temporal_series_index(
            self.A_img_paths, self.num_frames, self.frame_step
        )

    def _select_temporal_index_A(self, frame_step):
        if not self._random_temporal_frame_step_enabled():
            if len(self.frames_counts) == 1:  # single video mario
                if self.range_A == 0:
                    return 0
                return random.randint(0, self.range_A - 1)

            series_index = (
                self.vid_series_paths,
                self.frames_counts,
                self.cumulative_sums,
                self.available_frame_pool,
            )
            return select_temporal_start_from_series(self.A_img_paths, series_index)

        if len(self.vid_series_paths) == 1:
            return self._select_single_temporal_start(self.num_A, frame_step)

        series_index = build_temporal_series_index(
            self.A_img_paths, self.num_frames, frame_step
        )
        return select_temporal_start_from_series(self.A_img_paths, series_index)

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
        effective_frame_step = self._sample_temporal_frame_step()
        index_A = self._select_temporal_index_A(effective_frame_step)
        if index_A is None:
            return None

        images_A = []
        labels_A = []
        global_context_A = []
        A_label_clses = []
        ref_A_img_path = self.A_img_paths[index_A]
        ref_name_A = ref_A_img_path.split("/")[-1][: self.num_common_char]

        crop_size_min = (
            self.opt.data_online_creation_crop_size_A
            - self.opt.data_online_creation_crop_delta_A
        )
        crop_size_max = (
            self.opt.data_online_creation_crop_size_A
            + self.opt.data_online_creation_crop_delta_A
        )
        crop_size = random.randint(crop_size_min, crop_size_max)

        for i in range(self.num_frames):
            cur_index_A = index_A + i * effective_frame_step

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

                crop_coordinates = crop_image(
                    cur_A_img_path,
                    cur_A_label_path,
                    mask_delta=mask_delta_A,
                    mask_random_offset=self.opt.data_online_creation_mask_random_offset_A,
                    crop_delta=0,
                    mask_square=self.opt.data_online_creation_mask_square_A,
                    broaden_rect_aug=getattr(
                        self.opt, "data_online_creation_mask_broaden_rect_aug_A", False
                    ),
                    crop_dim=crop_size,
                    output_dim=self.opt.data_load_size,
                    context_pixels=self.opt.data_online_context_pixels,
                    load_size=self.opt.data_online_creation_load_size_A,
                    load_size_keep_ratio=getattr(
                        self.opt, "data_online_creation_load_size_keep_ratio_A", False
                    ),
                    get_crop_coordinates=True,
                    fixed_mask_size=self.opt.data_online_fixed_mask_size,
                    fixed_mask_size_model=getattr(
                        self.opt, "data_online_creation_mask_fixed_size_A", -1
                    ),
                    fixed_mask_min_unmasked_border_model=getattr(
                        self.opt, "data_online_creation_mask_min_unmasked_border_A", 4
                    ),
                    crop_center=True,
                )
                crop_result = crop_image(
                    cur_A_img_path,
                    cur_A_label_path,
                    mask_delta=mask_delta_A,
                    mask_random_offset=self.opt.data_online_creation_mask_random_offset_A,
                    crop_delta=0,
                    mask_square=self.opt.data_online_creation_mask_square_A,
                    broaden_rect_aug=getattr(
                        self.opt, "data_online_creation_mask_broaden_rect_aug_A", False
                    ),
                    crop_dim=crop_size,
                    output_dim=self.opt.data_load_size,
                    context_pixels=self.opt.data_online_context_pixels,
                    load_size=self.opt.data_online_creation_load_size_A,
                    load_size_keep_ratio=getattr(
                        self.opt, "data_online_creation_load_size_keep_ratio_A", False
                    ),
                    crop_coordinates=crop_coordinates,
                    fixed_mask_size=self.opt.data_online_fixed_mask_size,
                    fixed_mask_size_model=getattr(
                        self.opt, "data_online_creation_mask_fixed_size_A", -1
                    ),
                    fixed_mask_min_unmasked_border_model=getattr(
                        self.opt, "data_online_creation_mask_min_unmasked_border_A", 4
                    ),
                    crop_center=True,
                    return_meta=b2b_global_context_enabled_from_opt(self.opt),
                )
                if b2b_global_context_enabled_from_opt(self.opt):
                    (
                        cur_A_img,
                        cur_A_label,
                        ref_A_bbox,
                        A_ref_bbox_id,
                        crop_meta,
                    ) = crop_result
                    global_context_A.append(
                        build_masked_global_context_image(
                            cur_A_img_path,
                            crop_meta,
                            self.opt.data_online_creation_load_size_A,
                            getattr(
                                self.opt,
                                "data_online_creation_load_size_keep_ratio_A",
                                False,
                            ),
                        )
                    )
                else:
                    cur_A_img, cur_A_label, ref_A_bbox, A_ref_bbox_id = crop_result
                A_label_clses.append(int(ref_A_bbox[0]))
                images_A.append(cur_A_img)
                labels_A.append(cur_A_label)

                if i == 0:
                    A_ref_bbox = ref_A_bbox[1:]

            except Exception as e:
                print(e, f"{i+1}th frame of domain A in temporal dataloading")
                return None

        images_A, labels_A, A_ref_bbox = self.transform(images_A, labels_A, A_ref_bbox)
        if b2b_global_context_enabled_from_opt(self.opt):
            global_context_A = transform_global_context_images(
                self.opt,
                global_context_A,
                getattr(self.transform, "last_geometry_state", {}),
                getattr(self.opt, "alg_b2b_global_context_size", 128),
            )
        A_ref_label = labels_A[0]
        A_ref_img = images_A[0]
        images_A = torch.stack(images_A)
        labels_A = torch.stack(labels_A)
        A_label_clses = torch.tensor(A_label_clses, dtype=torch.long)
        A_label_clses = A_label_clses.to(images_A.device)

        result = {
            "A_ref": A_ref_img,
            "A": images_A,
            "A_img_paths": ref_A_img_path,
            "A_ref_bbox": A_ref_bbox,
            "A_label_mask": labels_A,
            "A_ref_label_mask": A_ref_label,
            "A_label_cls": A_label_clses,
            "A_temporal_frame_step": effective_frame_step,
            "B_ref": A_ref_img,
            "B": images_A,
            "B_img_paths": ref_A_img_path,
            "B_ref_bbox": A_ref_bbox,
            "B_label_mask": labels_A,
            "B_ref_label_mask": A_ref_label,
            "B_label_cls": A_label_clses,
            "B_temporal_frame_step": effective_frame_step,
            "temporal_frame_step": effective_frame_step,
        }
        if b2b_global_context_enabled_from_opt(self.opt):
            result["A_global_context"] = global_context_A
            result["B_global_context"] = global_context_A

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

        result["A_label_cls"] = A_label_clses
        result["B_label_cls"] = A_label_clses
        return result

    def sanitize(self):
        sanitized_A_img_paths = []
        sanitized_A_label_mask_paths = []
        for img_path, label_path in zip(self.A_img_paths, self.A_label_mask_paths):
            if os.path.exists(img_path) and os.path.exists(label_path):
                sanitized_A_img_paths.append(img_path)
                sanitized_A_label_mask_paths.append(label_path)
        self.A_img_paths = sanitized_A_img_paths
        self.A_label_mask_paths = sanitized_A_label_mask_paths
        if self.opt.data_max_dataset_size != float("inf"):
            self.A_img_paths, self.A_label_mask_paths = (
                self.A_img_paths[: self.opt.data_max_dataset_size],
                self.A_label_mask_paths[: self.opt.data_max_dataset_size],
            )
        self.A_size = len(self.A_img_paths)
        self.B_size = len(self.B_img_paths)
