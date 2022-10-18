import pytest
import torch.multiprocessing as mp
import sys

sys.path.append(sys.path[0] + "/..")
import train
from options.train_options import TrainOptions
from data import create_dataset

json_like_dict = {
    "name": "joligan_utest",
    "output_display_env": "joligan_utest",
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
    "G_unet_mha_inner_channel": 32,
    "G_unet_mha_num_head_channels": 16,
    "G_nblocks": 1,
    "G_padding_type": "reflect",
    "data_online_creation_rand_mask_A": True,
}


models_diffusion = ["palette"]

G_netG = ["unet_mha"]

D_proj_network_type = ["efficientnet", "vitsmall"]

f_s_net = ["unet", "segformer"]


def test_semantic_mask(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])
    for model in models_diffusion:
        json_like_dict["model_type"] = model
        json_like_dict["name"] += "_" + model
        json_like_dict_c = json_like_dict.copy()
        for Gtype in G_netG:
            json_like_dict_c["G_netG"] = Gtype
            opt = TrainOptions().parse_json(json_like_dict_c)
            train.launch_training(opt)
