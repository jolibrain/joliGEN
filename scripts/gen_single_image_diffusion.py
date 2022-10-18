import sys
import os
import json

sys.path.append("../")
from models import diffusion_networks
from options.train_options import TrainOptions
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import argparse
from data.online_creation import fill_mask_with_random, fill_mask_with_color


def load_model(modelpath, model_in_file, device):
    train_json_path = modelpath + "/train_config.json"
    with open(train_json_path, "r") as jsonf:
        train_json = json.load(jsonf)
    opt = TrainOptions().parse_json(train_json)
    if opt.model_multimodal:
        opt.model_input_nc += opt.train_mm_nz
    opt.jg_dir = "../"

    model = diffusion_networks.define_G(**vars(opt))
    model.eval()
    model.load_state_dict(torch.load(modelpath + "/" + model_in_file))

    model = model.to(device)
    return model, opt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-in-file", help="file path to generator model (.pth file)", required=True
)

parser.add_argument("--img-size", default=256, type=int, help="square image size")
parser.add_argument("--img-in", help="image to transform", required=True)
parser.add_argument(
    "--mask-in", help="mask used for image transformation", required=True
)
parser.add_argument("--img-out", help="transformed image", required=True)
parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
parser.add_argument("--gpuid", type=int, default=0, help="which GPU to use")
args = parser.parse_args()

# loading model
modelpath = args.model_in_file.replace(os.path.basename(args.model_in_file), "")
print("modelpath=", modelpath)

if not args.cpu:
    device = torch.device("cuda:" + str(args.gpuid))
model, opt = load_model(modelpath, os.path.basename(args.model_in_file), device)

# reading image
img = cv2.imread(args.img_in)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread(args.mask_in, 0)

# preprocessing
totensor = transforms.ToTensor()
resize = transforms.Resize(args.img_size)
tranlist = [
    totensor,
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    resize,
]
tran = transforms.Compose(tranlist)
img_tensor = tran(img).clone().detach()
mask = torch.from_numpy(np.array(mask, dtype=np.int64)).unsqueeze(0)
mask = resize(mask).clone().detach()

if not args.cpu:
    img_tensor = img_tensor.to(device).clone().detach()
    mask = mask.to(device).clone().detach()

if opt.data_online_creation_rand_mask_A:
    cond_image = fill_mask_with_random(
        img_tensor.clone().detach(), mask.clone().detach(), -1
    )
elif opt.data_online_creation_color_mask_A:
    cond_image = fill_mask_with_color(
        img_tensor.clone().detach(), mask.clone().detach(), {}
    )

# run through model
cond_image, img_tensor, mask = (
    cond_image.unsqueeze(0).clone().detach(),
    img_tensor.unsqueeze(0).clone().detach(),
    mask.unsqueeze(0).clone().detach(),
)


with torch.no_grad():
    out_tensor, visu = model.restoration(
        cond_image.clone().detach(),
        y_t=cond_image.clone().detach(),
        y_0=img_tensor.clone().detach(),
        mask=mask.clone().detach(),
        sample_num=2,
    )


print("outtensor", out_tensor.shape, "visu", visu.shape)

temp = img_tensor - out_tensor
print(temp.mean(), temp.min(), temp.max())
print(visu.shape)

# out_tensor = visu[-1:]

# post-processing
out_img = out_tensor.detach().data.cpu().float().numpy()[0]
img_np = img_tensor.detach().data.cpu().float().numpy()[0]
cond_image = cond_image.detach().data.cpu().float().numpy()[0]
# cond_image = torch.randn_like(cond_image)
visu = visu.detach().data.cpu().float().numpy()
visu1 = visu[1]
visu2 = visu[2]
visu0 = visu[0]

temp = out_img - img_np
print("np", temp.mean(), temp.min(), temp.max())

out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
img_np = (np.transpose(img_np, (1, 2, 0)) + 1) / 2.0 * 255.0
cond_image = (np.transpose(cond_image, (1, 2, 0)) + 1) / 2.0 * 255.0
visu0 = (np.transpose(visu0, (1, 2, 0)) + 1) / 2.0 * 255.0
visu1 = (np.transpose(visu1, (1, 2, 0)) + 1) / 2.0 * 255.0
visu2 = (np.transpose(visu2, (1, 2, 0)) + 1) / 2.0 * 255.0
print(out_img)
print(img_np)

temp = out_img - img_np
print("np", temp.mean(), temp.min(), temp.max())

out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(args.img_out, out_img)

img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
cv2.imwrite("/data1/pnsuau/checkpoints/test_palette_4/img_np.jpg", img_np)

cond_image = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("/data1/pnsuau/checkpoints/test_palette_4/cond_image.jpg", cond_image)


visu0 = cv2.cvtColor(visu0, cv2.COLOR_RGB2BGR)
cv2.imwrite("/data1/pnsuau/checkpoints/test_palette_4/visu0.jpg", visu0)

visu1 = cv2.cvtColor(visu1, cv2.COLOR_RGB2BGR)
cv2.imwrite("/data1/pnsuau/checkpoints/test_palette_4/visu1.jpg", visu1)

visu2 = cv2.cvtColor(visu2, cv2.COLOR_RGB2BGR)
cv2.imwrite("/data1/pnsuau/checkpoints/test_palette_4/visu2.jpg", visu2)


print("Successfully generated image ", args.img_out)
