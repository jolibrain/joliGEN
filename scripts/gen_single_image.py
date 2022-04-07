import sys
import os

sys.path.append("../")
from models import networks
from options.train_options import TrainOptions
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-in-file", help="file path to generator model (.pth file)", required=True
)
parser.add_argument(
    "--model-type",
    default="mobile_resnet_9blocks",
    help="model type, e.g. mobile_resnet_9blocks",
)
parser.add_argument(
    "--model-config",
    help="optional model configuration, e.g /path/to/segformer_config_b0.py",
)
parser.add_argument(
    "--padding-type",
    type=str,
    help="whether to use padding, zeros or reflect",
    default="reflect",
)
parser.add_argument("--img-size", default=256, type=int, help="square image size")
parser.add_argument("--img-in", help="image to transform", required=True)
parser.add_argument("--img-out", help="transformed image", required=True)
parser.add_argument("--bw", action="store_true", help="whether input/output is bw")
parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
args = parser.parse_args()

if args.bw:
    input_nc = output_nc = 1
else:
    input_nc = output_nc = 3

# loading model
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
    opt.G_config_segformer = (
        args.model_config
    )  # e.g. '/path/to/models/configs/segformer/segformer_config_b0.py'
opt.jg_dir = os.path.join("/".join(__file__.split("/")[:-2]))
model = networks.define_G(**vars(opt))

model.eval()
model.load_state_dict(torch.load(args.model_in_file))

if not args.cpu:
    model = model.cuda()

# reading image
img = cv2.imread(args.img_in)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# preprocessing
tranlist = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
tran = transforms.Compose(tranlist)
img_tensor = tran(img)
if not args.cpu:
    img_tensor = img_tensor.cuda()

# run through model
out_tensor = model(img_tensor.unsqueeze(0))[0].detach()

# post-processing
out_img = out_tensor.data.cpu().float().numpy()
print(out_img.shape)
out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
# print(out_img)
out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(args.img_out, out_img)
print("Successfully generated image ", args.img_out)
