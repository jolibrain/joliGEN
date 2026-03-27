import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.append(sys.path[0] + "/..")

import train
import models.mat_model as mat_model_module
from options.train_options import TrainOptions
from scripts.gen_video_mat import run_video_inference


json_like_dict = {
    "name": "joligen_utest_vid_mat_motion",
    "output_display_env": "joligen_utest_vid_mat_motion",
    "output_display_id": 0,
    "gpu_ids": "0",
    "model_type": "mat",
    "data_dataset_mode": "self_supervised_vid_mask_online",
    "data_load_size": 256,
    "data_crop_size": 256,
    "data_online_creation_crop_size_A": 256,
    "data_online_creation_crop_delta_A": 30,
    "data_online_creation_mask_delta_A_ratio": [[0.2, 0.2]],
    "data_online_creation_rand_mask_A": True,
    "data_temporal_number_frames": 2,
    "data_temporal_frame_step": 1,
    "alg_mat_motion": True,
    "alg_mat_motion_max_frames": 2,
    "alg_mat_motion_num_attention_heads": 4,
    "alg_mat_motion_num_transformer_blocks": 1,
    "alg_mat_pcp_ratio": 0.0,
    "train_n_epochs": 1,
    "train_n_epochs_decay": 0,
    "train_batch_size": 1,
    "test_batch_size": 1,
    "train_save_latest_freq": 1,
    "output_print_freq": 1000,
    "data_num_threads": 0,
    "data_max_dataset_size": 4,
    "data_relative_paths": True,
    "train_G_ema": True,
    "dataaug_no_rotate": True,
    "train_export_jit": False,
}


class TinyPerceptualLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, gt):
        return torch.mean(torch.abs(x - gt)), None


def test_mat_vid_online(dataroot, monkeypatch):
    monkeypatch.setattr(mat_model_module, "MATPerceptualLoss", TinyPerceptualLoss)

    json_like_dict_c = json_like_dict.copy()
    json_like_dict_c["dataroot"] = dataroot
    json_like_dict_c["checkpoints_dir"] = "/".join(dataroot.split("/")[:-1])

    opt = TrainOptions().parse_json(json_like_dict_c, save_config=True)
    train.launch_training(opt)

    model_in_file = os.path.join(
        json_like_dict_c["checkpoints_dir"],
        json_like_dict_c["name"],
        "latest_net_G_A.pth",
    )
    assert os.path.isfile(model_in_file)

    paths_in_file = os.path.join(json_like_dict_c["dataroot"], "trainA", "paths.txt")
    assert os.path.isfile(paths_in_file)

    infer_args = SimpleNamespace(
        name="mat_motion_vid_infer",
        model_in_file=model_in_file,
        dataroot=paths_in_file,
        dir_out=os.path.join(
            json_like_dict_c["checkpoints_dir"], json_like_dict_c["name"]
        ),
        data_prefix=json_like_dict_c["dataroot"],
        img_width=256,
        img_height=256,
        cpu=not torch.cuda.is_available(),
        gpuid=0,
        compare=False,
        fps=4,
        nb_img_max=2,
        sv_frames=False,
        start_frame=-1,
        bbox_ref_id=-1,
        motion_num_frames=2,
        motion_autoregressive=False,
        freeze_noise_across_frames=False,
    )
    video_path = run_video_inference(infer_args)
    assert os.path.isfile(video_path)
    assert os.path.getsize(video_path) > 0
