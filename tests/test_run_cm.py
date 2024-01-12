import os
import pytest
import torch.multiprocessing as mp
import sys
from itertools import product

sys.path.append(sys.path[0] + "/..")
import train
from options.train_options import TrainOptions
from data import create_dataset
from itertools import product

from scripts.gen_single_image_diffusion import InferenceDiffusionOptions, inference

json_like_dict = {
    "name": "joligen_utest",
    "output_display_env": "joligen_utest",
    "output_display_id": 0,
    "gpu_ids": "0",
    "data_dataset_mode": "self_supervised_labeled_mask",
    "data_load_size": 128,
    "data_crop_size": 128,
    "train_n_epochs": 1,
    "train_n_epochs_decay": 0,
    "data_max_dataset_size": 10,
    "data_relative_paths": True,
    "train_G_ema": True,
    "dataaug_no_rotate": True,
    "G_unet_mha_num_head_channels": 16,
    # "G_unet_mha_channel_mults": [1, 2],
    "G_diff_n_timestep_train": 50000,
    "G_nblocks": 1,
    "G_padding_type": "reflect",
    "data_online_creation_rand_mask_A": True,
}

infer_options = {
    "gpu_ids": "0",
    "img_width": 128,
    "img_height": 128,
}

models_diffusion = ["cm"]
G_netG = ["unet_mha"]

alg_diffusion_cond_embed = ["y_t"]

product_list = product(
    models_diffusion,
    G_netG,
    alg_diffusion_cond_embed,
)


def test_semantic_mask(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])

    with open(
        os.path.join(json_like_dict["dataroot"], "trainA", "paths.txt")
    ) as paths_file:
        line = paths_file.readline().strip().split(" ")
        infer_options["img_in"] = os.path.join(json_like_dict["dataroot"], line[0])
        infer_options["mask_in"] = os.path.join(json_like_dict["dataroot"], line[1])

    for (
        model,
        Gtype,
        alg_diffusion_cond_embed,
    ) in product_list:

        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["model_type"] = model
        json_like_dict_c["name"] += "_" + model
        json_like_dict_c["G_netG"] = Gtype
        json_like_dict_c["alg_diffusion_cond_embed"] = alg_diffusion_cond_embed

        opt = TrainOptions().parse_json(json_like_dict_c, save_config=True)
        train.launch_training(opt)

        # Inference
        infer_options_c = infer_options.copy()
        infer_options_c["model_in_file"] = os.path.join(
            json_like_dict_c["checkpoints_dir"],
            json_like_dict_c["name"],
            "latest_net_G_A.pth",
        )
        infer_options_c["dir_out"] = os.path.join(
            json_like_dict_c["checkpoints_dir"], json_like_dict_c["name"]
        )

        opt = InferenceDiffusionOptions().parse_json(infer_options_c, save_config=False)
        inference(opt)
