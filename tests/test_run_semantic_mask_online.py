import sys
import os
from itertools import product

import pytest
import torch.multiprocessing as mp

sys.path.append(sys.path[0] + "/..")
import train
from data import create_dataset
from options.train_options import TrainOptions
from scripts.gen_single_image import InferenceGANOptions, inference

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
    "data_online_creation_load_size_A": [1000, 2500],
    "data_online_creation_load_size_B": [1000, 2500],
    "data_online_context_pixels": 0,
    "train_n_epochs": 1,
    "train_n_epochs_decay": 0,
    "data_max_dataset_size": 10,
    "train_mask_out_mask": True,
    "f_s_net": "unet",
    "f_s_semantic_nclasses": 7,
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
    "data_temporal_number_frames": 4,
    "data_temporal_frame_step": 2,
    "train_semantic_mask": True,
    "train_temporal_criterion": False,
    "train_export_jit": True,
    "train_save_latest_freq": 10,
}

infer_options = {
    "gpu_ids": "0",
}

models_semantic_mask = [
    "cycle_gan",
    "cut",
]

G_netG = ["mobile_resnet_attn", "segformer_attn_conv"]

D_proj_network_type = ["efficientnet", "vitsmall", "dinov2_vits14"]

D_netDs = [
    ["basic", "projected_d"],
    ["basic", "temporal"],
    ["projected_d", "sam"],
]

f_s_net = ["unet"]

model_type_sam = ["mobile_sam"]

data_online_context_pixels = [0, 10]

with_amp = [False, True]

product_list = product(
    models_semantic_mask,
    G_netG,
    D_proj_network_type,
    D_netDs,
    f_s_net,
    model_type_sam,
    data_online_context_pixels,
    with_amp,
)


def test_semantic_mask_online(dataroot):
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
        Dtype,
        Dnet,
        f_s_type,
        sam_type,
        data_online_context_pixels,
        with_amp,
    ) in product_list:
        if model == "cycle_gan":
            if (
                "sam" in Dnet or "temporal" in Dnet
            ):  # cycle_gan+temporal does not fit in CI memory
                continue
        json_like_dict_c = json_like_dict.copy()
        json_like_dict_c["model_type"] = model
        json_like_dict_c["name"] += "_" + model
        json_like_dict_c["G_netG"] = Gtype
        json_like_dict_c["D_proj_network_type"] = Dtype
        json_like_dict_c["D_netDs"] = Dnet
        if "temporal" in Dnet:
            json_like_dict_c["data_dataset_mode"] = "temporal_labeled_mask_online"
        json_like_dict_c["f_s_net"] = f_s_type
        json_like_dict_c["model_type_sam"] = sam_type
        json_like_dict_c["data_online_context_pixels"] = data_online_context_pixels
        json_like_dict_c["with_amp"] = with_amp

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

        opt = InferenceGANOptions().parse_json(infer_options_c, save_config=True)
        inference(opt)
