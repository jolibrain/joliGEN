import argparse
import os
import sys

import torch

JG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(JG_DIR)

import b2b_onnx_denoiser_infer_autoregressive_progress_bbox as onnx_runner
from b2b_export_onnx import build_model, load_train_options, parse_device


def resolve_weights_path(model_in_file, use_ema):
    if not use_ema:
        return model_in_file

    if not model_in_file.endswith(".pth"):
        raise ValueError("--use_ema expects a .pth checkpoint path")

    ema_path = model_in_file[:-4] + "_ema.pth"
    if not os.path.isfile(ema_path):
        raise FileNotFoundError(f"EMA checkpoint not found: {ema_path}")
    return ema_path


def resolve_train_config(model_in_file, train_config):
    if train_config:
        return train_config
    return os.path.join(
        os.path.dirname(os.path.abspath(model_in_file)), "train_config.json"
    )


class PthDenoiserSession:
    """Small ONNX Runtime-compatible adapter around a JoliGEN B2B .pth model."""

    def __init__(self, model, device):
        self.model = model.b2b_model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def run(self, output_names, inputs):
        if output_names != ["output"]:
            raise ValueError(f"Expected output_names=['output'], got {output_names}")

        model_input = torch.from_numpy(inputs["model_input"]).to(
            self.device, dtype=torch.float32
        )
        timesteps = torch.from_numpy(inputs["timesteps"]).to(
            self.device, dtype=torch.float32
        )
        labels = torch.from_numpy(inputs["labels"]).to(self.device, dtype=torch.long)

        output = self.model(model_input, timesteps.flatten(), labels)
        return [output.detach().cpu().numpy()]


def load_pth_session(model_in_file, train_config, device, use_ema):
    train_config_path = resolve_train_config(model_in_file, train_config)
    if not os.path.isfile(train_config_path):
        raise FileNotFoundError(f"train_config.json not found: {train_config_path}")

    opt = load_train_options(train_config_path, device)
    weights_path = resolve_weights_path(model_in_file, use_ema)
    model = build_model(opt, weights_path, device)
    return PthDenoiserSession(model, device), weights_path, train_config_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_in_file",
        required=True,
        help="Path to a JoliGEN B2B checkpoint, e.g. latest_net_G_A.pth",
    )
    parser.add_argument(
        "--paths_in_file",
        required=True,
        help="paths.txt containing 'image_rel_path bbox_rel_path' pairs",
    )
    parser.add_argument(
        "--dataset_root",
        help="Dataset root used to resolve relative paths. Defaults to dirname(dirname(paths_in_file))",
    )
    parser.add_argument(
        "--train_config",
        help="Optional train_config.json for model, crop, mask, and denoising settings",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument(
        "--start_index", type=int, default=0, help="Start line index in paths.txt"
    )
    parser.add_argument(
        "--bbox_index", type=int, default=0, help="Which bbox line to use"
    )
    parser.add_argument("--label", type=int, default=None, help="Override class label")
    parser.add_argument("--seed", type=int, default=0, help="Seed for init_noise")
    parser.add_argument(
        "--denoise_steps",
        type=int,
        help="Override denoise step count. Defaults to first entry from alg.b2b_denoise_timesteps",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use the sibling *_ema.pth checkpoint instead of model_in_file",
    )
    parser.add_argument(
        "--single_step",
        action="store_true",
        help="Only process one 2-frame window",
    )
    parser.add_argument(
        "--debug_dump_dir",
        help="Optional directory to dump per-step denoiser inputs as .npy files",
    )
    parser.add_argument(
        "--autoregressive_reinject_patch",
        "--autoregressive-reinject-patch",
        action="store_true",
        help=(
            "Feed the previously generated crop back as known context in the next "
            "sliding window by replacing its y_t/y_0 tensors and zeroing its mask."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = parse_device(args.device)

    session, weights_path, train_config_path = load_pth_session(
        args.model_in_file, args.train_config, device, args.use_ema
    )
    train_json, _ = onnx_runner.load_train_config(
        args.model_in_file, train_config_path
    )

    train_frames, train_height, train_width, _, _ = onnx_runner.get_train_shape(
        train_json
    )
    if train_json.get("alg", {}).get("diffusion_cond_image_creation", "y_t") != "y_t":
        raise NotImplementedError(
            "This runner currently supports only alg.diffusion_cond_image_creation = 'y_t'."
        )

    dataset_root = onnx_runner.resolve_dataset_root(
        args.paths_in_file, args.dataset_root
    )
    pairs = onnx_runner.read_paths_file(args.paths_in_file)
    if args.start_index + 2 > len(pairs):
        raise ValueError(
            f"paths.txt has {len(pairs)} entries, need at least 2 from start_index={args.start_index}"
        )
    if args.single_step:
        pairs = pairs[args.start_index : args.start_index + 2]
    else:
        pairs = pairs[args.start_index :]

    denoise_steps = onnx_runner.resolve_denoise_steps(
        train_json, args.denoise_steps
    )
    frames_written = onnx_runner.run_sequence(
        session=session,
        pairs=pairs,
        dataset_root=dataset_root,
        output_dir=args.output_dir,
        bbox_index=args.bbox_index,
        label=args.label,
        seed=args.seed,
        train_json=train_json,
        denoise_steps=denoise_steps,
        debug_dump_dir=args.debug_dump_dir,
        autoregressive_reinject_patch=args.autoregressive_reinject_patch,
    )

    print(f"dataset_root : {dataset_root}")
    print(f"checkpoint   : {weights_path}")
    print(f"train_config : {train_config_path}")
    print(f"device       : {device}")
    print(f"train_shape  : {(train_frames, train_height, train_width)}")
    print(f"denoise_steps: {denoise_steps}")
    print(f"autoregressive_reinject_patch: {args.autoregressive_reinject_patch}")
    print(f"written      : {len(frames_written)} frames")
    print(f"saved        : {args.output_dir}")


if __name__ == "__main__":
    main()
