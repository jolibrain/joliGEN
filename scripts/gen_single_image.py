import json
import os
import sys
import logging

jg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(jg_dir)
from models import gan_networks
from options.train_options import TrainOptions
from options.inference_gan_options import InferenceGANOptions
import cv2
import numpy as np
import torch
import torchvision
from models import gan_networks
from options.train_options import TrainOptions
from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image
import tifffile
from util.util import rgbn_float_img_to_8bits_display


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
    opt.jg_dir = jg_dir

    if not cpu:
        device = torch.device("cuda:" + str(gpuid))
    else:
        device = torch.device("cpu")

    model = gan_networks.define_G(**vars(opt))
    model.eval()

    if (
        hasattr(model, "unet")
        and hasattr(model, "vae")
        and any("lora" in n for n, _ in model.unet.named_parameters())
    ):
        model.load_lora_config(modelpath + "/" + model_in_file)
    else:
        model.load_state_dict(
            torch.load(modelpath + "/" + model_in_file, map_location=device)
        )

    model = model.to(device)
    return model, opt, device


def inference_logger(name):

    PROCESS_NAME = "gen_single_image"
    LOG_PATH = os.environ.get(
        "LOG_PATH", os.path.join(os.path.dirname(__file__), "../logs")
    )
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(f"{LOG_PATH}/{name}.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(f"inference %s %s" % (PROCESS_NAME, name))


def inference(args):
    PROGRESS_NUM_STEPS = 6
    logger = inference_logger(args.name)
    logger.info(f"[1/%i] launch inference" % PROGRESS_NUM_STEPS)

    modelpath = os.path.dirname(args.model_in_file)
    print("modelpath=%s" % modelpath)
    model, opt, device = load_model(
        modelpath, os.path.basename(args.model_in_file), args.cpu, args.gpuid
    )
    logger.info(f"[2/%i] model loaded" % PROGRESS_NUM_STEPS)

    # reading image
    img_width = args.img_width if args.img_width is not None else opt.data_crop_size
    img_height = args.img_height if args.img_height is not None else opt.data_crop_size
    if opt.data_image_bits > 8 and opt.model_input_nc == 1:
        img = Image.open(args.img_in)
        # if img_height != opt.data_crop_size or img_width != opt.data_crop_size:
        #    print(
        #        "Requested image size differs from training crop size, resizing is not supported for images with more than 8 bits per channel"
        #    )
        #    exit(1)
    elif opt.data_image_bits > 8 and opt.model_input_nc > 1:
        img = tifffile.imread(args.img_in)
    else:
        img = cv2.imread(args.img_in)
    original_img = img.copy()
    if opt.data_image_bits == 8 and opt.model_input_nc != 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    logger.info(f"[3/%i] image loaded" % PROGRESS_NUM_STEPS)

    # preprocessing
    if opt.data_image_bits > 8:
        tranlist = [transforms.ToTensor()]
        tranlist += [torchvision.transforms.v2.ToDtype(torch.float32)]
        bit_scaling = 2**opt.data_image_bits - 1
        tranlist += [transforms.Lambda(lambda img: img * (1 / float(bit_scaling)))]
        tranlist += [
            transforms.Normalize((0.5,), (0.5,))
        ]  # XXX: > 8bit, mono canal only for now
    else:
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

    logger.info(f"[4/%i] preprocessing finished" % PROGRESS_NUM_STEPS)

    # run through model
    if opt.G_netG == "img2img_turbo":
        out_tensor = model(img_tensor, args.prompt)[0].detach()
    else:
        out_tensor = model(img_tensor)[0].detach()

    logger.info(f"[5/%i] out tensor available" % PROGRESS_NUM_STEPS)

    # post-processing
    out_img = out_tensor.data.cpu().float().numpy()
    print(out_img.shape)

    # print(out_img)
    if opt.data_image_bits > 8 and opt.model_input_nc > 1:
        out_img = np.transpose(out_img, (1, 2, 0)) + 1
        rgb_img, nrg_img = rgbn_float_img_to_8bits_display(out_img, gamma=0.7)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        nrg_img = cv2.cvtColor(nrg_img, cv2.COLOR_RGB2BGR)
        out_img = np.concatenate((rgb_img, nrg_img), axis=1)
    else:
        out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    if args.compare:
        original_img_cv = np.array(original_img)
        np.multiply(original_img_cv, 255.0 / 4096.0, original_img_cv, casting="unsafe")
        original_img_cv = cv2.cvtColor(original_img_cv, cv2.COLOR_RGB2BGR)
        print("original image size=", original_img_cv.shape)
        print("generated image size=", out_img.shape)
        out_img = np.concatenate((original_img_cv, out_img), axis=1)

    cv2.imwrite(args.img_out, out_img)

    logger.info(f"[6/%i] success - %s" % (PROGRESS_NUM_STEPS, args.img_out))
    print("Successfully generated image ", args.img_out)


if __name__ == "__main__":
    opt = InferenceGANOptions().parse(save_config=False)
    inference(opt)
