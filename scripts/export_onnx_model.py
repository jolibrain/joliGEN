import sys
import torch
import numpy as np

sys.path.append("../")
from models import gan_networks
from options.train_options import TrainOptions
import argparse
import os
import json

from mmcv.onnx import register_extra_symbolics

opset_version = 9

parser = argparse.ArgumentParser()
parser.add_argument("--train-config", help="Path to train_config.json")
parser.add_argument(
    "--model-in-file",
    help="file path to generator model to export (.pth file)",
    required=True,
)
parser.add_argument("--model-out-file", help="file path to exported model (.onnx file)")
parser.add_argument(
    "--model-type",
    default="mobile_resnet_9blocks",
    help="model type, e.g. mobile_resnet_9blocks",
)
parser.add_argument(
    "--model-config",
    help="optional model configuration, e.g /path/to/segformer_config_b0.py",
)
parser.add_argument("--img-size", default=256, type=int, help="square image size")
parser.add_argument("--cpu", action="store_true", help="whether to export for CPU")
parser.add_argument("--bw", action="store_true", help="whether input/output is bw")
parser.add_argument(
    "--padding-type",
    type=str,
    help="whether to use padding, zeros or reflect",
    default="zeros",
)
parser.add_argument("--opset-version", type=int, default=9, help="ONNX opset version")
args = parser.parse_args()

if not args.model_out_file:
    model_out_file = args.model_in_file.replace(".pth", ".onnx")
else:
    model_out_file = args.model_out_file

if args.bw:
    input_nc = output_nc = 1
else:
    input_nc = output_nc = 3

if args.train_config:
    with open(args.train_config) as train_config_file:
        train_config_json = json.load(train_config_file)
        opt = TrainOptions().parse_json(train_config_json)
else:
    opt = TrainOptions().parse_json({})

opt.data_crop_size = args.img_size
opt.data_load_size = args.img_size
opt.G_attn_nb_mask_attn = 10
opt.G_attn_nb_mask_input = 1
opt.G_netG = args.model_type
opt.G_padding_type = args.padding_type
opt.model_input_nc = input_nc
opt.model_output_nc = output_nc

if "segformer" in args.model_type:
    args.opset_version = 11  # enforce opset 11
    register_extra_symbolics(args.opset_version)
    opt.G_config_segformer = (
        args.model_config
    )  # e.g. '/path/to/models/configs/segformer/segformer_config_b0.py'
opt.jg_dir = os.path.join("/".join(__file__.split("/")[:-2]))
model = gan_networks.define_G(**vars(opt))

model.eval()
model.load_state_dict(torch.load(args.model_in_file))

if not args.cpu:
    model = model.cuda()

# export to ONNX via tracing
if args.cpu:
    device = "cpu"
else:
    device = "cuda"
dummy_input = torch.randn(1, input_nc, args.img_size, args.img_size, device=device)


torch.onnx.export(
    model, dummy_input, model_out_file, verbose=True, opset_version=args.opset_version
)
