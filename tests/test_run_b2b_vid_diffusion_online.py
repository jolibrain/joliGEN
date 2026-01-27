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
import torch
from scripts.gen_vid_diffusion import InferenceDiffusionOptions, inference


json_like_dict = {
    "name": "joligen_utest_vid",
    "output_display_env": "joligen_utest_vid",
    "output_display_id": 0,
    "gpu_ids": "0",
    "data_load_size": 256,
    "data_crop_size": 256,
    "data_online_creation_crop_size_A": 300,
    "data_online_creation_crop_delta_A": 30,
    "data_online_creation_mask_delta_A_ratio": [[0.2, 0.2]],
    "train_n_epochs": 1,
    "train_n_epochs_decay": 0,
    "data_max_dataset_size": 10,
    "data_relative_paths": True,
    "train_G_ema": True,
    "dataaug_no_rotate": True,
    "data_online_creation_rand_mask_A": True,
    "train_export_jit": False,
    "train_save_latest_freq": 10,
    "train_batch_size": 1,
    "data_temporal_number_frames": 8,
    "data_temporal_frame_step": 1,
    "alg_b2b_denoise_timesteps": [1],
}


G_netG = ["vit_vid"]
alg_diffusion_cond_embed = ["y_t"]
models_diffusion = ["b2b"]
data_dataset_mode = ["self_supervised_vid_mask_online"]

infer_options = {
    "gpu_ids": "0",
    "img_width": 256,
    "img_height": 256,
}
product_list = product(
    models_diffusion,
    G_netG,
    alg_diffusion_cond_embed,
)


def test_sc_model(dataroot):
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

        # cuda is available
        if not torch.cuda.is_available():
            infer_options_c["cpu"] = True

        opt = InferenceDiffusionOptions().parse_json(infer_options_c, save_config=False)
        inference(opt)
