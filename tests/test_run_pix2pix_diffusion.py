import pytest
import torch.multiprocessing as mp
import sys
from itertools import product

sys.path.append(sys.path[0] + "/..")
import train
from options.train_options import TrainOptions
from data import create_dataset

json_like_dict = {
    "name": "joligen_utest",
    "output_display_env": "joligen_utest",
    "output_display_id": 0,
    "gpu_ids": "0",
    "data_dataset_mode": "aligned",
    "data_load_size": 256,
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
    "alg_diffusion_task": "pix2pix",
    "dataaug_flip": "both",
}


models_diffusion = ["palette"]
G_netG = ["unet_mha", "uvit"]
G_efficient = [True]
G_unet_mha_norm_layer = [
    "groupnorm",
]
product_list = product(models_diffusion, G_netG, G_efficient, G_unet_mha_norm_layer)


def test_diffusion(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])
    for model, Gtype, G_efficient, G_norm in product_list:
        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["model_type"] = model
        json_like_dict_c["name"] += "_" + model
        json_like_dict_c["G_netG"] = Gtype
        json_like_dict_c["G_unet_mha_norm_layer"] = G_norm
        json_like_dict_c["G_unet_mha_vit_efficient"] = G_efficient

        opt = TrainOptions().parse_json(json_like_dict_c, save_config=True)
        train.launch_training(opt)
