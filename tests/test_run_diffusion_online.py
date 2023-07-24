import sys
from itertools import product

import pytest
import torch.multiprocessing as mp

sys.path.append(sys.path[0] + "/..")
import train
from data import create_dataset
from options.train_options import TrainOptions

json_like_dict = {
    "name": "joligen_utest",
    "output_display_env": "joligen_utest",
    "output_display_id": 0,
    "gpu_ids": "0",
    "data_load_size": 128,
    "data_crop_size": 128,
    "data_online_creation_crop_size_A": 420,
    "data_online_creation_crop_delta_A": 50,
    "data_online_creation_mask_delta_A": [[50, 50]],
    "data_online_creation_mask_delta_A_ratio": [[0.2, 0.2]],
    "data_online_creation_crop_size_B": 420,
    "data_online_creation_crop_delta_B": 50,
    "data_online_creation_load_size_A": [2500, 1000],
    "data_online_creation_load_size_B": [2500, 1000],
    "train_n_epochs": 1,
    "train_n_epochs_decay": 0,
    "data_max_dataset_size": 10,
    "data_relative_paths": True,
    "train_G_ema": True,
    "dataaug_no_rotate": True,
    "G_unet_mha_inner_channel": 32,
    "G_unet_mha_num_head_channels": 16,
    "G_unet_mha_channel_mults": [1, 2],
    "G_nblocks": 1,
    "G_padding_type": "reflect",
    "data_online_creation_rand_mask_A": True,
    "train_export_jit": True,
    "train_save_latest_freq": 10,
}


models_diffusion = ["palette"]
G_netG = ["unet_mha", "uvit"]
data_dataset_mode = ["self_supervised_labeled_mask_online", "self_supervised_temporal"]

product_list = product(models_diffusion, G_netG, data_dataset_mode)


def test_diffusion_online(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])
    for model, Gtype, dataset_mode in product_list:
        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["model_type"] = model
        json_like_dict_c["name"] += "_" + model
        json_like_dict_c["G_netG"] = Gtype

        json_like_dict_c["data_dataset_mode"] = dataset_mode
        if dataset_mode == "self_supervised_temporal":
            json_like_dict_c["data_temporal_number_frames"] = 2
            json_like_dict_c["data_temporal_frame_step"] = 1
            json_like_dict_c["data_temporal_num_common_char"] = 3
            json_like_dict_c["alg_palette_cond_image_creation"] = "previous_frame"

        opt = TrainOptions().parse_json(json_like_dict_c)
        train.launch_training(opt)
