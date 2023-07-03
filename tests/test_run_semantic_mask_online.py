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
    "G_netG": "mobile_resnet_attn",
    "output_display_env": "joligen_utest",
    "output_display_id": 0,
    "gpu_ids": "0",
    "data_dataset_mode": "unaligned_labeled_mask_online",
    "data_load_size": 180,
    "data_crop_size": 180,
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
    "train_mask_out_mask": True,
    "f_s_net": "unet",
    "f_s_semantic_nclasses": 2,
    "dataaug_D_noise": 0.001,
    "train_sem_use_label_B": True,
    "data_relative_paths": True,
    "D_netDs": ["basic", "projected_d", "temporal"],
    "D_weight_sam": "models/configs/sam/pretrain/mobile_sam.pt",
    "train_gan_mode": "projected",
    "D_proj_interp": 256,
    "train_G_ema": True,
    "dataaug_no_rotate": True,
    "train_mask_compute_miou": True,
    "train_mask_miou_every": 1,
    "data_temporal_number_frames": 2,
    "data_temporal_frame_step": 2,
    "train_semantic_mask": True,
    "train_temporal_criterion": True,
    "train_export_jit": True,
    "train_save_latest_freq": 10,
}

models_semantic_mask = [
    "cycle_gan",
    "cut",
]

G_netG = ["mobile_resnet_attn", "segformer_attn_conv"]

D_proj_network_type = ["efficientnet", "vitsmall"]

D_netDs = [["basic", "projected_d", "temporal"], ["sam"]]

f_s_net = ["unet"]

model_type_sam = ["mobile_sam"]

product_list = product(
    models_semantic_mask, G_netG, D_proj_network_type, D_netDs, f_s_net, model_type_sam
)


def test_semantic_mask_online(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])

    for model, Gtype, Dtype, Dnet, f_s_type, sam_type in product_list:
        if model == "cycle_gan" and "sam" in Dnet:
            continue
        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["model_type"] = model
        json_like_dict_c["name"] += "_" + model
        json_like_dict_c["G_netG"] = Gtype
        json_like_dict_c["D_proj_network_type"] = Dtype
        json_like_dict_c["D_netDs"] = Dnet
        json_like_dict_c["f_s_net"] = f_s_type
        json_like_dict_c["model_type_sam"] = sam_type

        opt = TrainOptions().parse_json(json_like_dict_c)
        train.launch_training(opt)
