import sys
import torch
import numpy as np
sys.path.append('../')
from models import networks
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-in-file',help='file path to generator model to export (.pth file)',required=True)
parser.add_argument('--model-out-file',help='file path to exported model (.pt file)')
parser.add_argument('--model-type',default='mobile_resnet_9blocks',help='model type, e.g. mobile_resnet_9blocks')
parser.add_argument('--img-size',default=256,type=int,help='square image size')
args = parser.parse_args()

if not args.model_out_file:
    model_out_file = args.model_in_file.replace('.pth','.pt')
else:
    model_out_file = args.model_out_file

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

jit_model = torch.jit.script(model)
jit_model.save(model_out_file)
