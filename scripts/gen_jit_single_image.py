import sys
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_in_file",
    help="file path to generator model to export (.pt file)",
    required=True,
)
parser.add_argument("--img_in", help="image to transform", required=True)
parser.add_argument("--img_out", help="transformed image", required=True)
parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
parser.add_argument("--img_size", default=256, type=int, help="square image size")
args = parser.parse_args()

model = torch.jit.load(args.model_in_file)
if not args.cpu:
    model = model.cuda()

# reading image
img = cv2.imread(args.img_in)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (args.img_size, args.img_size))

# preprocessing
tranlist = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
tran = transforms.Compose(tranlist)
img_tensor = tran(img)
if not args.cpu:
    img_tensor = img_tensor.cuda()
# print('tensor shape=',img_tensor.shape)

# run through model
out_tensor = model(img_tensor.unsqueeze(0))[0].detach()
# print(out_tensor)
# print(out_tensor.shape)

# post-processing
out_img = out_tensor.data.cpu().float().numpy()
out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(args.img_out, out_img)
print("Successfully generated image ", args.img_out)
