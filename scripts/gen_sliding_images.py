import sys
import os
import glob

sys.path.append("../")
from models import networks
from options.train_options import TrainOptions
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import argparse


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])


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
parser.add_argument(
    "--img-in", help="image or image folder to transform", required=True
)
parser.add_argument(
    "--stepsize",
    type=int,
    default=128,
    help="sliding window stepsize, to be set to image input size",
)
parser.add_argument("--windowsize", type=int, default=256, help="window input size")
parser.add_argument(
    "--output-dir", help="full size pictures output directory", required=True
)
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

if os.path.isfile(args.img_in):
    images = [args.img_in]
else:
    images = glob.glob(args.img_in + "*.*")

# preprocessing transforms
tranlist = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
tran = transforms.Compose(tranlist)

for image in images:

    # reading image
    img = cv2.imread(args.img_in)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(image, " / shape=", img.shape)

    outputmap = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    # - walk through sliding windows
    i = 0
    for (x, y, window) in sliding_window(
        img, stepSize=args.stepsize, windowSize=(args.windowsize, args.windowsize)
    ):

        # - if window is smaller than input sizes, fill it up correctly
        windowtmp = window.copy()
        resized = False
        if window.shape[0] != args.stepsize or window.shape[1] != args.stepsize:
            resized = True
            windowfull = np.zeros((args.windowsize, args.windowsize, 3), np.uint8)
            windowfull[0 : window.shape[0], 0 : window.shape[1]] = window.copy()
            window = windowfull

        # - get the local image window
        # windowpath = '/tmp/img'+str(i)+'.png'
        # cv2.imwrite(windowpath, window)

        # preprocessing
        img_tensor = tran(window)
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

        # - combine the output images
        if resized:
            out_img = out_img[0 : windowtmp.shape[0], 0 : windowtmp.shape[1]]
        outputmap[y : y + window.shape[1], x : x + window.shape[0]] = out_img

    i += 1

    # save the full size output image
    imgoutpath = (
        args.output_dir + "/" + os.path.basename(image).replace(".jpg", "") + "_gan.jpg"
    )
    cv2.imwrite(imgoutpath, outputmap)
    print("Successfully generated image ", imgoutpath)
