import sys
import os
from itertools import product

import pytest
import torch.multiprocessing as mp

sys.path.append(sys.path[0] + "/..")
import train
from data import create_dataset_temporal
from options.train_options import TrainOptions
from scripts.gen_vid_diffusion import InferenceDiffusionOptions, inference

json_like_dict = {
    "name": "joligen_utest_vid",
    "output_display_env": "joligen_utest_vid",
    "output_display_id": 0,
    "gpu_ids": "0",
    "data_load_size": 64,
    "data_crop_size": 64,
    "data_online_creation_crop_size_A": 64,
    "data_online_creation_crop_delta_A": 50,
    "data_online_creation_mask_delta_A_ratio": [[0.2, 0.2]],
    "data_online_creation_crop_size_B": 64,
    "data_online_creation_crop_delta_B": 50,
    "train_n_epochs": 1,
    "train_n_epochs_decay": 0,
    "data_max_dataset_size": 10,
    "data_relative_paths": True,
    "train_G_ema": True,
    "dataaug_no_rotate": True,
    "G_unet_mha_num_head_channels": 32,
    "G_unet_mha_channel_mults": [1, 2, 4, 8],
    "G_nblocks": 9,
    "G_padding_type": "reflect",
    "data_online_creation_rand_mask_A": True,
    "train_export_jit": False,
    "train_save_latest_freq": 10,
    "G_diff_n_timestep_test": 4,
    "train_batch_size": 1,
    "data_temporal_number_frames": 8,
    "data_temporal_frame_step": 1,
    "G_diff_n_timestep_train": 6,
}

infer_options = {
    "gpu_ids": "0",
    "img_width": 128,
    "img_height": 128,
    "alg_diffusion_cond_image_creation": "y_t",
    "model_type": "palette",
    "G_diff_n_timestep_test": 4,
}

models_diffusion = ["palette"]
G_netG = ["unet_vid"]
data_dataset_mode = ["self_supervised_vid_mask_online"]

product_list = product(models_diffusion, G_netG, data_dataset_mode)


def test_vid_diffusion_online(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])
    with open(
        os.path.join(
            json_like_dict["checkpoints_dir"],
            dataroot.split("/")[-1],
            "trainA",
            "paths.txt",
        )
    ) as paths_file:
        line = paths_file.readline().strip().split(" ")
        infer_options["img_in"] = os.path.join(json_like_dict["dataroot"], line[0])
        infer_options["mask_in"] = os.path.join(json_like_dict["dataroot"], line[1])

    for model, Gtype, dataset_mode in product_list:
        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["model_type"] = model
        json_like_dict_c["name"] += "_" + model
        json_like_dict_c["G_netG"] = Gtype
        json_like_dict_c["data_dataset_mode"] = dataset_mode
        opt = TrainOptions().parse_json(json_like_dict_c, save_config=True)
        train.launch_training(opt)

        # Inference
        infer_options_c = infer_options.copy()
        infer_options_c["model_in_file"] = os.path.join(
            json_like_dict_c["checkpoints_dir"],
            json_like_dict_c["name"],
            "latest_net_G_A.pth",
        )
        infer_options_c["paths_in_file"] = os.path.join(
            json_like_dict["dataroot"], "trainA", "paths.txt"
        )
        infer_options_c["dir_out"] = os.path.join(
            json_like_dict_c["checkpoints_dir"], json_like_dict_c["name"]
        )
        opt = InferenceDiffusionOptions().parse_json(infer_options_c, save_config=False)
        inference(opt)
