import pytest
import torch.multiprocessing as mp
import sys
from itertools import product
import os

sys.path.append(sys.path[0] + "/..")
import train
from options.train_options import TrainOptions
from data import create_dataset
from scripts.gen_single_image import InferenceGANOptions, inference

json_like_dict = {
    "name": "joligen_utest",
    "output_display_env": "joligen_utest",
    "output_display_id": 0,
    "gpu_ids": "0",
    "data_dataset_mode": "unaligned",
    "data_load_size": 128,
    "data_crop_size": 128,
    "train_batch_size": 1,
    "train_n_epochs": 1,
    "data_max_dataset_size": 10,
    "data_relative_paths": True,
    "G_nblocks": 1,
    "G_padding_type": "reflect",
    "dataaug_flip": "both",
    "D_netDs": ["basic"],
    "train_iter_size": 1,
    "G_prompt": "zebra",
    "G_lora_unet": 8,
    "G_lora_vea": 8,
    "train_n_epochs_decay": 0,
    "train_save_latest_freq": 10,
}

infer_options = {
    "gpu_ids": "0",
}
models_gan = ["cut"]
G_netG = ["img2img_turbo"]
G_efficient = [True]
G_unet_mha_norm_layer = [
    "groupnorm",
]
product_list = product(models_gan, G_netG, G_efficient, G_unet_mha_norm_layer)


def test_img2img_turbo(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])
    infer_options["img_in"] = os.path.join(
        json_like_dict["dataroot"], "trainA", "n02381460_9223.jpg"
    )
    for model, Gtype, G_efficient, G_norm in product_list:
        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["model_type"] = model
        json_like_dict_c["name"] += "_" + model
        json_like_dict_c["G_netG"] = Gtype
        json_like_dict_c["G_unet_mha_norm_layer"] = G_norm
        json_like_dict_c["G_unet_mha_vit_efficient"] = G_efficient
        json_like_dict_c["alg_cut_supervised_loss"] = ["L1"]

        opt = TrainOptions().parse_json(json_like_dict_c, save_config=True)
        train.launch_training(opt)

        # Inference
        infer_options_c = infer_options.copy()
        infer_options_c["model_in_file"] = os.path.join(
            json_like_dict_c["checkpoints_dir"],
            json_like_dict_c["name"],
            "latest_net_G_A.pth",
        )
        infer_options_c["img_out"] = os.path.join(
            json_like_dict_c["checkpoints_dir"],
            json_like_dict_c["name"],
            "test_image.jpg",
        )

        opt = InferenceGANOptions().parse_json(infer_options_c, save_config=False)
        inference(opt)
