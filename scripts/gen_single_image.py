import sys
sys.path.append('../')
from models import networks
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import argparse

#image_path = '/data2/clients/cartier/models/cartier_phase2_basic_mobilenet/input_patches/IMG_0050803.jpg'

parser = argparse.ArgumentParser()
parser.add_argument('--model-in-file',help='file path to generator model to export (.pth file)',required=True)
parser.add_argument('--model-type',default='mobile_resnet_9blocks',help='model type, e.g. mobile_resnet_9blocks')
parser.add_argument('--img-size',default=256,type=int,help='square image size')
parser.add_argument('--img-in',help='image to transform',required=True)
parser.add_argument('--img-out',help='transformed image',required=True)
parser.add_argument('--cpu',action='store_true',help='whether to use CPU')
args = parser.parse_args()

# loading model
input_nc = 3
output_nc = 3
ngf = 64
use_dropout = False
decoder = True
img_size = args.img_size
model = networks.define_G(input_nc,output_nc,ngf,args.model_type,'instance',use_dropout,
                          decoder=decoder,
                          img_size=args.img_size,
                          img_size_dec=args.img_size)
model.eval()
model.load_state_dict(torch.load(args.model_in_file))
if not args.cpu:
    model = model.cuda()

# reading image
img = cv2.imread(args.img_in)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# preprocessing
tranlist = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
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
#print(out_img)
out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
cv2.imwrite(args.img_out,out_img)
print('Successfully generated image ',args.img_out)
