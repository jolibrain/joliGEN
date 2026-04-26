import copy
import json
import random

import torch

from data.base_dataset import BaseDataset


ALLOWED_CHILD_DATASET_MODES = {"self_supervised_vid_mask_online"}

ALLOWED_CHILD_OVERRIDES = {
    "dataroot",
    "data_online_creation_crop_size_A",
    "data_online_creation_crop_delta_A",
    "data_online_creation_load_size_A",
    "data_online_creation_mask_delta_A",
    "data_online_creation_mask_delta_A_ratio",
    "data_online_creation_mask_random_offset_A",
    "data_online_creation_mask_square_A",
    "data_temporal_num_common_char",
}

FORBIDDEN_CHILD_OVERRIDES = {
    "data_load_size",
    "data_crop_size",
    "model_input_nc",
    "model_output_nc",
    "data_temporal_number_frames",
    "data_temporal_frame_step",
    "G_netG",
    "G_vit_num_classes",
    "model_type",
    "alg_b2b_mask_as_channel",
    "alg_diffusion_cond_image_creation",
}


class MultiDatasetDataset(BaseDataset):
    """Dataset wrapper that samples items from several child datasets."""

    def __init__(self, opt, phase, name=""):
        if phase not in {"train", "test"}:
            raise ValueError("multi_dataset is currently supported for train/test only")
        if not opt.data_multi_dataset_config:
            raise ValueError("--data_multi_dataset_config is required for multi_dataset")

        BaseDataset.__init__(self, opt, phase, name)

        self.max_retries = 20
        self.children = []
        self.child_names = []
        self.child_indices = []
        self.child_weights = []
        self.child_lengths = []
        self.reference_sample = None

        with open(opt.data_multi_dataset_config, "r") as config_file:
            config = json.load(config_file)

        entries = self._entries_for_phase(config, name)

        for dataset_index, entry in enumerate(entries):
            child_dataset, child_name, child_weight = self._build_child_dataset(
                entry, dataset_index
            )
            child_length = len(child_dataset)
            if child_length <= 0:
                raise ValueError(f"child dataset '{child_name}' is empty")

            sample = self._sample_from_child(child_dataset, child_length)
            if sample is None:
                raise ValueError(
                    f"child dataset '{child_name}' did not return a valid dry sample"
                )
            self._validate_sample(sample, child_name)

            self.children.append(child_dataset)
            self.child_names.append(child_name)
            self.child_indices.append(entry.get("_dataset_index", dataset_index))
            self.child_weights.append(child_weight)
            self.child_lengths.append(child_length)

        if sum(self.child_weights) <= 0:
            raise ValueError("at least one multi_dataset child weight must be > 0")

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __len__(self):
        return sum(self.child_lengths)

    def __getitem__(self, index):
        if self.phase == "test":
            return self._get_test_item(index)

        for _ in range(self.max_retries):
            dataset_index = random.choices(
                range(len(self.children)), weights=self.child_weights, k=1
            )[0]
            child = self.children[dataset_index]
            child_length = self.child_lengths[dataset_index]
            child_sample = child[random.randint(0, child_length - 1)]
            if child_sample is None:
                continue

            sample = dict(child_sample)
            sample["dataset_name"] = self.child_names[dataset_index]
            sample["dataset_index"] = dataset_index
            return sample

        return None

    def _get_test_item(self, index):
        child = self.children[0]
        child_length = self.child_lengths[0]
        sample = child[index % child_length]
        if sample is None:
            return None
        sample = dict(sample)
        sample["dataset_name"] = self.child_names[0]
        sample["dataset_index"] = self.child_indices[0]
        sample["dataset_test_name"] = self.name
        return sample

    def _entries_for_phase(self, config, name):
        datasets = config.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("multi_dataset config must contain a non-empty datasets list")

        if self.phase == "train":
            return datasets

        test_sets = config.get("test_sets")
        if not isinstance(test_sets, list) or not test_sets:
            raise ValueError(
                "multi_dataset config must contain a non-empty test_sets list "
                "for test phase"
            )

        for test_set in test_sets:
            if test_set.get("id") == name:
                return [self._entry_for_test_set(datasets, test_set)]

        available = [test_set.get("id") for test_set in test_sets]
        raise ValueError(
            f"unknown multi_dataset test set '{name}', available test sets: {available}"
        )

    def _entry_for_test_set(self, datasets, test_set):
        dataset_name = test_set.get("dataset_name")
        for dataset_index, dataset_entry in enumerate(datasets):
            if dataset_entry.get("name") == dataset_name:
                entry = copy.deepcopy(dataset_entry)
                entry["dataroot"] = test_set.get("dataroot", entry.get("dataroot"))
                entry["name"] = dataset_name
                entry["_dataset_index"] = dataset_index
                entry["_child_test_name"] = test_set.get("child_test_name", "")
                return entry

        raise ValueError(
            f"multi_dataset test set '{test_set.get('id')}' references unknown "
            f"dataset '{dataset_name}'"
        )

    def _build_child_dataset(self, entry, dataset_index):
        from data import find_dataset_using_name

        if not isinstance(entry, dict):
            raise ValueError(f"dataset entry {dataset_index} must be an object")

        child_name = entry.get("name", f"dataset_{dataset_index}")
        child_mode = entry.get("dataset_mode", "self_supervised_vid_mask_online")
        if child_mode not in ALLOWED_CHILD_DATASET_MODES:
            raise ValueError(
                f"unsupported child dataset_mode '{child_mode}' for '{child_name}'"
            )

        overrides = copy.deepcopy(entry.get("overrides", {}))
        if "dataroot" in entry:
            overrides["dataroot"] = entry["dataroot"]

        self._validate_overrides(overrides, child_name)

        child_opt = copy.deepcopy(self.opt)
        child_opt.data_dataset_mode = child_mode
        for key, value in overrides.items():
            setattr(child_opt, key, value)

        child_class = find_dataset_using_name(child_mode)
        child_name_arg = (
            entry.get("_child_test_name", "") if self.phase == "test" else self.name
        )
        child_dataset = child_class(child_opt, self.phase, child_name_arg)
        child_weight = float(entry.get("weight", 1.0))
        if child_weight < 0:
            raise ValueError(f"child dataset '{child_name}' has a negative weight")

        return child_dataset, child_name, child_weight

    def _validate_overrides(self, overrides, child_name):
        for key in overrides:
            if key in FORBIDDEN_CHILD_OVERRIDES:
                raise ValueError(
                    f"child dataset '{child_name}' cannot override shape/model option "
                    f"'{key}'"
                )
            if key not in ALLOWED_CHILD_OVERRIDES:
                raise ValueError(
                    f"child dataset '{child_name}' uses unsupported override '{key}'"
                )
            if not hasattr(self.opt, key):
                raise ValueError(
                    f"child dataset '{child_name}' override '{key}' is not a known option"
                )

    def _sample_from_child(self, child_dataset, child_length):
        for sample_index in range(min(self.max_retries, child_length)):
            sample = child_dataset[sample_index % child_length]
            if sample is not None:
                return sample
        for _ in range(self.max_retries):
            sample = child_dataset[random.randint(0, child_length - 1)]
            if sample is not None:
                return sample
        return None

    def _validate_sample(self, sample, child_name):
        if not isinstance(sample, dict):
            raise ValueError(f"child dataset '{child_name}' must return dictionaries")

        signature = self._sample_signature(sample)
        if self.reference_sample is None:
            self.reference_sample = signature
            return

        if signature != self.reference_sample:
            raise ValueError(
                f"child dataset '{child_name}' returns keys or tensor shapes that do "
                "not match the first child dataset"
            )

    def _sample_signature(self, sample):
        signature = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                signature[key] = ("tensor", tuple(value.shape), value.dtype)
            else:
                signature[key] = (type(value).__name__,)
        return signature
