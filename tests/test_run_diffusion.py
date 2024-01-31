import pytest
import torch.multiprocessing as mp
import sys
from itertools import product

sys.path.append(sys.path[0] + "/..")
import train
from options.train_options import TrainOptions
from data import create_dataset
from itertools import product

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
    "G_unet_mha_channel_mults": [1, 2],
    "G_nblocks": 1,
    "G_padding_type": "reflect",
    "data_online_creation_rand_mask_A": True,
    "f_s_semantic_nclasses": 2,
}

models_diffusion = ["palette"]
G_netG = ["unet_mha", "uvit", "hdit"]
G_efficient = [True, False]
train_feat_wavelet = [False, True]

alg_diffusion_cond_embed = [
    "y_t"  # , "mask"
]  # , "class"] class conditioning can't be tested for now because there is no class in the dataset

product_list = product(
    models_diffusion,
    G_netG,
    alg_diffusion_cond_embed,
    G_efficient,
    train_feat_wavelet,
)


def test_semantic_mask(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])

    for (
        model,
        Gtype,
        alg_diffusion_cond_embed,
        G_efficient,
        train_feat_wavelet,
    ) in product_list:
        if train_feat_wavelet and Gtype == "hdit":
            continue
        if G_efficient and Gtype == "hdit":
            continue

        print(
            "Testing",
            model,
            Gtype,
            alg_diffusion_cond_embed,
            G_efficient,
            train_feat_wavelet,
        )

        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["model_type"] = model
        json_like_dict_c["name"] += "_" + model
        json_like_dict_c["G_netG"] = Gtype

        json_like_dict_c["G_unet_mha_vit_efficient"] = G_efficient

        json_like_dict_c["alg_diffusion_cond_embed"] = alg_diffusion_cond_embed
        json_like_dict_c["train_feat_wavelet"] = train_feat_wavelet

        opt = TrainOptions().parse_json(json_like_dict_c, save_config=True)
        train.launch_training(opt)
