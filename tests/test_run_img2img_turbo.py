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
    "data_load_size": 48,
    "data_crop_size": 48,
    "train_batch_size": 1,
    "train_n_epochs": 1,
    "data_max_dataset_size": 10,
    "data_relative_paths": True,
    "G_nblocks": 1,
    "G_padding_type": "reflect",
    "dataaug_flip": "both",
    "D_netDs": ["vision_aided"],
    "train_iter_size": 1,
    "G_prompt": "zebra",
    "G_lora_unet": 2,
    "G_lora_vae": 2,
    "train_n_epochs_decay": 0,
    "train_save_latest_freq": 10,
}

infer_options = {
    "gpu_ids": "0",
}
models_gan = ["cut"]
G_netG = ["img2img_turbo"]
product_list = product(models_gan, G_netG)


def test_img2img_turbo(dataroot):
    json_like_dict["dataroot"] = dataroot
    json_like_dict["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])
    infer_options["img_in"] = os.path.join(
        json_like_dict["dataroot"], "trainA", "n02381460_9223.jpg"
    )
    for model, Gtype in product_list:
        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["model_type"] = model
        json_like_dict_c["name"] += "_" + model
        json_like_dict_c["G_netG"] = Gtype

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
