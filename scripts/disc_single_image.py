import sys

sys.path.append("../")
import os
from models import networks
import cv2
import torch
from torchvision import transforms
from options.base_options import BaseOptions
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-in-file",
    help="file path to discriminator model (.pth file)",
    required=True,
)
parser.add_argument(
    "--model-type", default="projected_d", help="model type, e.g. projected_d, basic"
)
parser.add_argument("--img-size", default=256, type=int, help="square image size")
parser.add_argument("--img-in", help="image to transform", required=True)
parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
args = parser.parse_args()

opt = BaseOptions().parse_json({})
opt.model_input_nc = 3
opt.model_output_nc = 3
opt.D_netD = args.model_type
opt.D_ndf = 64
opt.D_dropout = False
opt.data_crop_size = args.img_size
opt.jg_dir = os.path.join("/".join(__file__.split("/")[:-2]))
model = networks.define_D(netD=opt.D_netD, **vars(opt))
model.eval()

# loading model
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
out_vec = out_tensor.data.cpu().float().numpy()
# print(out_vec.shape)
# print(out_vec)
print(out_vec.mean())
