import json
import logging
import os
import sys
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

jg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(jg_dir)

from data.online_creation import crop_image, fill_mask_with_color, fill_mask_with_random
from models import create_model
from options.train_options import TrainOptions


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key[len("module.") :]: value for key, value in state_dict.items()}


def load_mat_model(model_in_file, cpu, gpuid):
    model_dir = os.path.dirname(model_in_file)
    train_json_path = os.path.join(model_dir, "train_config.json")
    with open(train_json_path, "r") as jsonf:
        train_json = json.load(jsonf)

    train_json["gpu_ids"] = "-1" if cpu else str(gpuid)
    opt = TrainOptions().parse_json(train_json, save_config=False, set_device=False)
    opt.isTrain = False
    opt.phase = "test"
    opt.use_cuda = (not cpu) and torch.cuda.is_available() and len(opt.gpu_ids) > 0

    if opt.model_type != "mat":
        raise ValueError(
            f"Expected a MAT checkpoint, got model_type={opt.model_type!r}"
        )

    rank = 0
    device = torch.device(f"cuda:{gpuid}" if opt.use_cuda else "cpu")
    model = create_model(opt, rank)

    state_dict = torch.load(model_in_file, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = strip_module_prefix(state_dict)

    model.netG_A.load_state_dict(state_dict, strict=True)
    model.netG_A = model.netG_A.to(device)
    model.eval()

    return model, opt, device


def inference_logger(name):
    process_name = "gen_single_image_mat"
    log_path = os.environ.get("LOG_PATH", os.path.join(jg_dir, "logs"))
    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(
                os.path.join(log_path, f"{name}.log"),
                mode="w",
            ),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(f"inference {process_name} {name}")


def load_rgb_image(img_path, img_width, img_height):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((img_width, img_height), Image.BICUBIC)
    original_np = np.array(image)
    image_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )(image)
    return original_np, image_tensor


def load_label_mask(mask_path, img_width, img_height):
    mask = Image.open(mask_path)
    mask = mask.resize((img_width, img_height), Image.NEAREST)
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    mask_tensor = torch.from_numpy(mask_np).to(dtype=torch.float32).unsqueeze(0)
    return mask_np, mask_tensor


def resolve_mask_delta(opt):
    mask_delta_ratio = getattr(opt, "data_online_creation_mask_delta_A_ratio", [[]])
    if mask_delta_ratio != [[]]:
        return mask_delta_ratio
    return getattr(opt, "data_online_creation_mask_delta_A", [[]])


def load_bbox_mask_crop(opt, img_path, bbox_path, img_width, img_height, bbox_ref_id):
    if img_width != img_height:
        raise ValueError("MAT bbox inference requires square output dimensions")

    mask_random_offset = getattr(
        opt, "data_online_creation_mask_random_offset_A", [0.0]
    )
    if mask_random_offset != [0.0]:
        warnings.warn(
            "disabling data_online_creation_mask_random_offset_A in MAT inference mode"
        )
        mask_random_offset = [0.0]

    image, mask, _, _ = crop_image(
        img_path=img_path,
        bbox_path=bbox_path,
        mask_random_offset=mask_random_offset,
        mask_delta=resolve_mask_delta(opt),
        crop_delta=0,
        mask_square=getattr(opt, "data_online_creation_mask_square_A", False),
        crop_dim=getattr(opt, "data_online_creation_crop_size_A", img_width),
        output_dim=img_width,
        context_pixels=getattr(opt, "data_online_context_pixels", 0),
        load_size=getattr(opt, "data_online_creation_load_size_A", []),
        select_cat=getattr(opt, "data_online_select_category", -1),
        crop_center=True,
        fixed_mask_size=getattr(opt, "data_online_fixed_mask_size", -1),
        bbox_ref_id=bbox_ref_id,
        inverted_mask=getattr(opt, "data_inverted_mask", False),
        single_bbox=getattr(opt, "data_online_single_bbox", False),
        random_bbox=False,
    )

    original_np = np.array(image)
    image_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )(image)

    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    mask_tensor = torch.from_numpy(mask_np).to(dtype=torch.float32).unsqueeze(0)

    return original_np, image_tensor, mask_tensor


def create_masked_input(opt, clean_tensor, mask_tensor):
    if getattr(opt, "data_online_creation_rand_mask_A", False):
        return fill_mask_with_random(clean_tensor, mask_tensor, -1)
    if getattr(opt, "data_online_creation_color_mask_A", False):
        return fill_mask_with_color(clean_tensor, mask_tensor, {})
    raise ValueError(
        "MAT inference requires either data_online_creation_rand_mask_A or "
        "data_online_creation_color_mask_A to be enabled in train_config.json"
    )


def prepare_mat_batch(
    opt, img_in, mask_in, img_width, img_height, device, bbox_ref_id=-1
):
    if mask_in.endswith(".txt"):
        original_np, clean_tensor, mask_tensor = load_bbox_mask_crop(
            opt, img_in, mask_in, img_width, img_height, bbox_ref_id
        )
    else:
        original_np, clean_tensor = load_rgb_image(img_in, img_width, img_height)
        _, mask_tensor = load_label_mask(mask_in, img_width, img_height)
    masked_tensor = create_masked_input(opt, clean_tensor, mask_tensor)

    clean_tensor = clean_tensor.unsqueeze(0).to(device)
    masked_tensor = masked_tensor.unsqueeze(0).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)

    batch = {
        "A": masked_tensor,
        "B": clean_tensor,
        "A_img_paths": [img_in],
        "B_img_paths": [img_in],
        "A_label_mask": mask_tensor,
        "B_label_mask": mask_tensor.clone(),
    }

    return batch, original_np, masked_tensor[0].detach().cpu()


def tensor_to_bgr_uint8(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]
    image = tensor.detach().cpu().float().numpy()
    image = (np.transpose(image, (1, 2, 0)) + 1.0) / 2.0
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def compose_inference_frame(output_tensor, original_np, masked_tensor, compare):
    output_bgr = tensor_to_bgr_uint8(output_tensor)
    if compare:
        original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
        masked_bgr = tensor_to_bgr_uint8(masked_tensor)
        output_bgr = np.concatenate([original_bgr, masked_bgr, output_bgr], axis=1)
    return output_bgr


def save_inference_image(output_tensor, original_np, masked_tensor, img_out, compare):
    output_bgr = compose_inference_frame(
        output_tensor, original_np, masked_tensor, compare
    )
    out_dir = os.path.dirname(img_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(img_out, output_bgr)


def inference(args):
    progress_num_steps = 5
    logger = inference_logger(args.name)
    logger.info("[1/%i] launch inference", progress_num_steps)

    model, opt, device = load_mat_model(args.model_in_file, args.cpu, args.gpuid)
    logger.info("[2/%i] model loaded", progress_num_steps)

    img_width = args.img_width if args.img_width is not None else opt.data_crop_size
    img_height = args.img_height if args.img_height is not None else opt.data_crop_size

    batch, original_np, masked_tensor = prepare_mat_batch(
        opt,
        args.img_in,
        args.mask_in,
        img_width,
        img_height,
        device,
        args.bbox_ref_id,
    )
    logger.info("[3/%i] inputs prepared", progress_num_steps)

    model.set_input(batch)
    with torch.inference_mode():
        model.inference(1)
    logger.info("[4/%i] model inference complete", progress_num_steps)

    save_inference_image(
        model.fake_B, original_np, masked_tensor, args.img_out, args.compare
    )
    logger.info("[5/%i] success - %s", progress_num_steps, args.img_out)


if __name__ == "__main__":
    from options.inference_mat_options import InferenceMATOptions

    options = InferenceMATOptions().parse(save_config=False)
    inference(options)
