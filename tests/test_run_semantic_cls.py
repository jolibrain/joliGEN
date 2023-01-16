import pytest
import torch.multiprocessing as mp
import sys
from itertools import product

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
    "data_dataset_mode": "unaligned_labeled_cls",
    "data_load_size": 180,
    "data_crop_size": 180,
    "train_n_epochs": 1,
    "train_n_epochs_decay": 0,
    "data_max_dataset_size": 10,
    "train_mask_out_mask": True,
    "cls_semantic_nclasses": 5,
    "dataaug_D_noise": 0.001,
    "train_sem_use_label_B": True,
    "D_netDs": ["basic", "projected_d"],
    "train_gan_mode": "projected",
    "D_proj_interp": 256,
    "train_G_ema": True,
    "dataaug_no_rotate": True,
    "train_semantic_cls": True,
}

models_semantic_mask = [
    "cut",
    "cycle_gan",
]

G_netG = ["mobile_resnet_attn", "segformer_attn_conv"]

D_proj_network_type = ["efficientnet", "vitsmall"]

f_s_net = ["unet", "segformer"]

product_list = product(models_semantic_mask, G_netG, D_proj_network_type, f_s_net)


def test_semantic_mask(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])

    for model, Gtype, Dtype, f_s_type in product_list:
        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["model_type"] = model
        json_like_dict_c["G_netG"] = Gtype
        json_like_dict_c["D_proj_network_type"] = Dtype
        json_like_dict_c["f_s_net"] = f_s_type

        opt = TrainOptions().parse_json(json_like_dict_c)
        train.launch_training(opt)
