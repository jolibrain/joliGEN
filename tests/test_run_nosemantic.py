import pytest
import torch.multiprocessing as mp
import sys
sys.path.append(sys.path[0]+"/..")
import train
from options.train_options import TrainOptions
from data import create_dataset

json_like_dict={
    'name': 'joligan_utest',
    'G_netG': 'mobile_resnet_attn',
    'output_display_env': 'joligan_utest',
    'output_display_id': 0,
    'gpu_ids': '0',
    'data_dataset_mode': 'unaligned',
    'data_load_size': 180,
    'data_crop_size': 180,
    'train_n_epochs': 1,
    'train_n_epochs_decay': 0,
    'data_max_dataset_size': 10,
}

models_nosemantic=[
    "cycle_gan",
    "cut",
]

def test_nosemantic(dataroot):
    json_like_dict['dataroot']=dataroot
    json_like_dict['checkpoints_dir']="/".join(dataroot.split("/")[:-1])
    for model in models_nosemantic:
        json_like_dict['model_type'] = model
        json_like_dict['name'] += '_' + model
        opt = TrainOptions().parse_json(json_like_dict.copy())
        train.launch_training(opt)

