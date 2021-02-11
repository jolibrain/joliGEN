import sys
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-in-file',help='file path to generator model to export (.pt file)',required=True)
parser.add_argument('--img-in',help='image to transform',required=True)
parser.add_argument('--img-out',help='transformed image',required=True)
args = parser.parse_args()

model = torch.jit.load(args.model_in_file)

# reading image
img = cv2.imread(args.img_in)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# preprocessing
tranlist = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
tran = transforms.Compose(tranlist)
img_tensor = tran(img)
#print(img_tensor)

# run through model
out_tensor = model(img_tensor.unsqueeze(0))[0].detach()
#print(out_tensor)
#print(out_tensor.shape)

# post-processing
out_img = out_tensor.data[0].cpu().float().numpy()
out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
#print(out_img)
out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
#print(out_img)
cv2.imwrite(args.img_out,out_img)
print('Successfully generated image ',args.img_out)
