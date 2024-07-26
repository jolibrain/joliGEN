import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.append("../")

from models import gan_networks
from options.train_options import TrainOptions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_in_file",
        help="file path to generator model (.pth file)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--video_in", help="video to transform", type=Path, required=True
    )
    parser.add_argument(
        "--video_out", help="transformed video", type=Path, required=True
    )
    parser.add_argument(
        "--img_width", type=int, help="image width, defaults to model crop size"
    )
    parser.add_argument(
        "--img_height", type=int, help="image height, defaults to model crop size"
    )
    parser.add_argument(
        "--max_frames", type=int, help="Select total number of frames to generate"
    )
    parser.add_argument("--fps", type=int, help="select FPS")
    parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
    parser.add_argument("--gpuid", type=int, default=0, help="which GPU to use")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="put the input video on the left side to compare",
    )
    parser.add_argument(
        "--n_inferences",
        type=int,
        default=1,
        help="Number of recursive inferences per frame",
    )
    return parser.parse_args()


def get_z_random(
    batch_size: int = 1, nz: int = 8, random_type: str = "gauss"
) -> torch.Tensor:
    if random_type == "uni":
        z = torch.rand(batch_size, nz) * 2.0 - 1.0
    elif random_type == "gauss":
        z = torch.randn(batch_size, nz)
    return z.detach()


def iter_video_frames(video_path: Path, max_frames: int) -> np.ndarray:
    """Iterate over frames in a video."""
    cap = cv2.VideoCapture(str(video_path))
    max_frames = min(max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    for _ in tqdm(range(max_frames), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def preprocess_frame(
    frame: np.ndarray, img_width: int, img_height: int, transforms: transforms.Compose
) -> torch.Tensor:
    """Preprocess a single frame."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    frame = transforms(frame)
    return frame


def postprocess_frame(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().cpu().float().numpy()
    frame = np.transpose(frame, (1, 2, 0))
    frame = (frame + 1) / 2.0 * 255.0
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = frame.astype(np.uint8)
    return frame


def load_model(model_dir: Path, model_filename: Path, device: torch.device):
    train_json_path = model_dir / "train_config.json"
    with open(train_json_path, "r") as jsonf:
        train_json = json.load(jsonf)
    opt = TrainOptions().parse_json(train_json, set_device=False)
    if opt.model_multimodal:
        opt.model_input_nc += opt.train_mm_nz
    opt.jg_dir = "../"

    model = gan_networks.define_G(**vars(opt))
    model.eval()
    model.load_state_dict(torch.load(model_dir / model_filename, map_location=device))

    model = model.to(device)
    return model, opt


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpuid}")

    # Load the model.
    model_dir = args.model_in_file.parent
    print(f"Model directory {model_dir}.")
    model, opt = load_model(model_dir, args.model_in_file.name, device)

    img_width = args.img_width if args.img_width is not None else opt.data_crop_size
    img_height = args.img_height if args.img_height is not None else opt.data_crop_size
    transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    video_width = img_width * 2 if args.compare else img_width
    video_height = img_height
    video_writer = cv2.VideoWriter(
        str(args.video_out),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        args.fps,
        (video_width, video_height),
    )

    # Optional noise.
    # Noise is sampled only once. The same noise is used for all video frames.
    if opt.model_multimodal:
        z_random = get_z_random(batch_size=1, nz=opt.train_mm_nz)
        z_random = z_random.to(device)

    with torch.inference_mode():
        for frame in iter_video_frames(args.video_in, args.max_frames):
            original_frame = frame.copy()
            for _ in range(args.n_inferences):
                frame = preprocess_frame(frame, img_width, img_height, transforms)
                frame = frame.to(device)
                frame = frame.unsqueeze(0)

                if opt.model_multimodal:
                    z_real = z_random.view(z_random.size(0), z_random.size(1), 1, 1)
                    z_real = z_real.expand(
                        z_random.size(0), z_random.size(1), frame.size(2), frame.size(3)
                    )
                    frame = torch.cat((frame, z_real), dim=1)

                frame = model(frame)[0]
                frame = postprocess_frame(frame)

            if args.compare:
                frame = np.concatenate((original_frame, frame), axis=1)

            video_writer.write(frame)

    print(f"Saving video to {args.video_out}.")
    video_writer.release()
