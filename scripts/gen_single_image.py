import sys
import os
import json
import logging

sys.path.append("../")
from models import gan_networks
from options.train_options import TrainOptions
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import argparse


def get_z_random(batch_size=1, nz=8, random_type="gauss"):
    if random_type == "uni":
        z = torch.rand(batch_size, nz) * 2.0 - 1.0
    elif random_type == "gauss":
        z = torch.randn(batch_size, nz)
    return z.detach()


def load_model(modelpath, model_in_file, cpu, gpuid):
    train_json_path = modelpath + "/train_config.json"
    with open(train_json_path, "r") as jsonf:
        train_json = json.load(jsonf)
    opt = TrainOptions().parse_json(train_json, set_device=False)
    if opt.model_multimodal:
        opt.model_input_nc += opt.train_mm_nz
    opt.jg_dir = "../"

    if not cpu:
        device = torch.device("cuda:" + str(gpuid))
    else:
        device = torch.device("cpu")

    model = gan_networks.define_G(**vars(opt))
    model.eval()
    model.load_state_dict(
        torch.load(modelpath + "/" + model_in_file, map_location=device)
    )

    model = model.to(device)
    return model, opt, device


def launch_predict_single_image(args, process_name):

    PROGRESS_NUM_STEPS = 6

    LOG_PATH = os.environ.get(
        "LOG_PATH", os.path.join(os.path.dirname(__file__), "../logs")
    )
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    log_file = f"{LOG_PATH}/{process_name}.log"

    logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO)
    logging.info(f"[1/%i] launch process" % PROGRESS_NUM_STEPS)

    # loading model
    modelpath = args.model_in_file.replace(os.path.basename(args.model_in_file), "")
    print("modelpath=", modelpath)
    logging.debug("modelpath=%s" % modelpath)

    model, opt, device = load_model(
        modelpath, os.path.basename(args.model_in_file), args.cpu, args.gpuid
    )
    logging.info(f"[2/%i] model loaded" % PROGRESS_NUM_STEPS)

    # reading image
    img = cv2.imread(args.img_in)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logging.info(f"[3/%i] image loaded" % PROGRESS_NUM_STEPS)

    # preprocessing
    tranlist = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    tran = transforms.Compose(tranlist)
    img_tensor = tran(img)
    if not args.cpu:
        img_tensor = img_tensor.to(device)

    if opt.model_multimodal:
        z_random = get_z_random(batch_size=1, nz=opt.train_mm_nz)
        z_random = z_random.to(device)
        # print('z_random shape=', self.z_random.shape)
        z_real = z_random.view(z_random.size(0), z_random.size(1), 1, 1).expand(
            z_random.size(0),
            z_random.size(1),
            img_tensor.size(1),
            img_tensor.size(2),
        )
        img_tensor = torch.cat([img_tensor.unsqueeze(0), z_real], 1)
    else:
        img_tensor = img_tensor.unsqueeze(0)
    logging.info(f"[4/%i] preprocessing finished" % PROGRESS_NUM_STEPS)

    # run through model
    out_tensor = model(img_tensor)[0].detach()
    logging.info(f"[5/%i] out tensor available" % PROGRESS_NUM_STEPS)

    # post-processing
    out_img = out_tensor.data.cpu().float().numpy()
    print(out_img.shape)
    logging.debug("out_img.shape=%s" % str(out_img.shape))
    out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    # print(out_img)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.img_out, out_img)
    print("Successfully generated image ", args.img_out)
    logging.info(f"[6/%i] success - %s" % (PROGRESS_NUM_STEPS, args.img_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-in-file",
        help="file path to generator model (.pth file)",
        required=True,
    )

    parser.add_argument("--img-in", help="image to transform", required=True)
    parser.add_argument("--img-out", help="transformed image", required=True)
    parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
    parser.add_argument("--gpuid", type=int, default=0, help="which GPU to use")
    args = parser.parse_args()

    launch_predict_single_image(args)
