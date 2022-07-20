import pytest
import torch.multiprocessing as mp
import sys

sys.path.append(sys.path[0] + "/..")
import train
from options.train_options import TrainOptions
from data import create_dataset

json_like_dict = {
    "name": "joligan_utest",
    "G_netG": "mobile_resnet_attn",
    "output_display_env": "joligan_utest",
    "output_display_id": 0,
    "gpu_ids": "0",
    "data_dataset_mode": "unaligned_labeled_mask",
    "data_load_size": 180,
    "data_crop_size": 180,
    "train_n_epochs": 1,
    "train_n_epochs_decay": 0,
    "data_max_dataset_size": 10,
    "train_mask_out_mask": True,
    "f_s_net": "unet",
    "f_s_semantic_nclasses": 2,
    "dataaug_D_noise": 0.001,
    "train_sem_use_label_B": True,
    "data_relative_paths": True,
    "D_netDs": ["basic", "projected_d"],
    "train_gan_mode": "projected",
    "D_proj_interp": 256,
    "train_G_ema": True,
    "dataaug_no_rotate": True,
    "train_mask_compute_miou": True,
    "train_mask_miou_every": 1000,
}

models_semantic_mask = [
    "cycle_gan_semantic_mask",
    "cut_semantic_mask",
]

G_netG = ["mobile_resnet_attn", "segformer_attn_conv"]

D_proj_network_type = ["efficientnet", "vitsmall"]

f_s_net = ["unet", "segformer"]


def test_semantic_mask(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])
    for model in models_semantic_mask:
        json_like_dict["model_type"] = model
        json_like_dict["name"] += "_" + model
        json_like_dict_c = json_like_dict.copy()
        for Gtype in G_netG:
            json_like_dict_c["G_netG"] = Gtype
            for Dtype in D_proj_network_type:
                json_like_dict_c["D_proj_network_type"] = Dtype
                for f_s_type in f_s_net:
                    json_like_dict_c["f_s_net"] = f_s_type
                    opt = TrainOptions().parse_json(json_like_dict_c)
                    train.launch_training(opt)
