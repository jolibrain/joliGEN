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
    "G_netG": "mobile_resnet_attn",
    "output_display_env": "joligen_utest",
    "output_display_id": 0,
    "gpu_ids": "0",
    "data_dataset_mode": "unaligned",
    "data_load_size": 128,
    "data_crop_size": 128,
    "train_n_epochs": 1,
    "train_n_epochs_decay": 0,
    "data_max_dataset_size": 10,
    "model_depth_network": "MiDaS_small",
    "train_export_jit": True,
    "train_save_latest_freq": 10,
    "dataaug_APA": False,
    "alg_cut_supervised_loss": ["L1", "LPIPS", "DISTS"],
    "alg_cut_lambda_supervised": 1.0,
    "alg_cut_lambda_perceptual": 0.3,
}

models = [
    "cut",
]

G_netGs = ["unet_mha", "hdit", "hat"]
D_netDs = [["basic"], ["none"]]

product_list = product(models, G_netGs, D_netDs)


def test_sr_gan(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])

    for model, Gtype, Dtype in product_list:
        json_like_dict["model_type"] = model
        json_like_dict["G_netGs"] = Gtype
        json_like_dict["D_netDs"] = Dtype

        opt = TrainOptions().parse_json(json_like_dict.copy(), save_config=True)
        train.launch_training(opt)
