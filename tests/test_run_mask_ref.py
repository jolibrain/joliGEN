import pytest
import torch.multiprocessing as mp
import sys
from itertools import product

sys.path.append(sys.path[0] + "/..")
import train
from options.train_options import TrainOptions
from data import create_dataset

json_like_dict = {
    "name": "joligen_utest_mask_ref",
    "output_display_env": "joligen_utest_mask_ref",
    "output_display_id": 0,
    "gpu_ids": [0],
    "data_load_size": 128,
    "data_crop_size": 128,
    "train_n_epochs": 1,
    "train_n_epochs_decay": 0,
    "data_max_dataset_size": 10,
    "data_relative_paths": True,
    "train_G_ema": True,
    "dataaug_no_rotate": True,
    "G_unet_mha_num_head_channels": 16,
    "G_unet_mha_channel_mults": [1, 2],
    "G_nblocks": 1,
    "G_padding_type": "reflect",
    "data_online_creation_rand_mask_A": True,
    "f_s_semantic_nclasses": 100,
    "model_type": "palette",
    "G_netG": "unet_mha",
}

models_datasets = [
    ["palette", "self_supervised_labeled_mask_ref"],
    ["cut", "unaligned_labeled_mask_ref"],
]
conditionings = [
    "alg_palette_conditioning",
    "alg_palette_cond_image_creation",
]

product_list = product(
    models_datasets,
    conditionings,
)


def test_mask_ref(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])

    for (model, dataset), conditioning in product_list:
        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["data_dataset_mode"] = dataset
        json_like_dict_c["model_type"] = model
        json_like_dict_c[conditioning] = "ref"
        opt = TrainOptions().parse_json(json_like_dict_c, save_config=True)
        train.launch_training(opt)
