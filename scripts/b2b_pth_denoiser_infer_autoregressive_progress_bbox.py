import argparse
import os
import sys

import torch

JG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(JG_DIR)

try:
    from scripts import (
        b2b_onnx_denoiser_infer_autoregressive_progress_bbox as onnx_runner,
    )
    from scripts.b2b_export_onnx import build_model, load_train_options, parse_device
except ImportError:
    import b2b_onnx_denoiser_infer_autoregressive_progress_bbox as onnx_runner
    from b2b_export_onnx import build_model, load_train_options, parse_device
from util.b2b_context import b2b_global_context_enabled_from_opt


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

    def __init__(
        self,
        model,
        device,
        mask_size_conditioning=False,
        mask_prediction=False,
        temporal_frame_step_conditioning=False,
        global_context_conditioning=False,
        object_ref_conditioning=False,
    ):
        self.model = model.b2b_model
        self.device = device
        self.mask_size_conditioning = mask_size_conditioning
        self.mask_prediction = mask_prediction
        self.temporal_frame_step_conditioning = temporal_frame_step_conditioning
        self.global_context_conditioning = global_context_conditioning
        self.object_ref_conditioning = object_ref_conditioning
        self.model.eval()

    @torch.no_grad()
    def run(self, output_names, inputs):
        available_outputs = (
            ["output", "mask_logits"] if self.mask_prediction else ["output"]
        )
        if any(name not in available_outputs for name in output_names):
            raise ValueError(
                f"Expected output names from {available_outputs}, got {output_names}"
            )

        model_input = torch.from_numpy(inputs["model_input"]).to(
            self.device, dtype=torch.float32
        )
        timesteps = torch.from_numpy(inputs["timesteps"]).to(
            self.device, dtype=torch.float32
        )
        labels = torch.from_numpy(inputs["labels"]).to(self.device, dtype=torch.long)

        model_kwargs = {}
        if self.mask_size_conditioning:
            if "mask_size_cond" not in inputs:
                raise ValueError("mask_size_cond input is required by this checkpoint")
            model_kwargs["mask_size_cond"] = torch.from_numpy(
                inputs["mask_size_cond"]
            ).to(self.device, dtype=torch.float32)
        if self.mask_prediction:
            if "mask_precision_mode" not in inputs:
                raise ValueError(
                    "mask_precision_mode input is required by this checkpoint"
                )
            if "mask_precision_severity" not in inputs:
                raise ValueError(
                    "mask_precision_severity input is required by this checkpoint"
                )
            model_kwargs["mask_precision_mode"] = torch.from_numpy(
                inputs["mask_precision_mode"]
            ).to(self.device, dtype=torch.long)
            model_kwargs["mask_precision_severity"] = torch.from_numpy(
                inputs["mask_precision_severity"]
            ).to(self.device, dtype=torch.float32)
        if self.temporal_frame_step_conditioning:
            if "temporal_frame_step" not in inputs:
                raise ValueError(
                    "temporal_frame_step input is required by this checkpoint"
                )
            model_kwargs["temporal_frame_step"] = torch.from_numpy(
                inputs["temporal_frame_step"]
            ).to(self.device, dtype=torch.float32)
        if self.global_context_conditioning:
            if "global_context" not in inputs:
                raise ValueError("global_context input is required by this checkpoint")
            model_kwargs["global_context"] = torch.from_numpy(
                inputs["global_context"]
            ).to(
                self.device,
                dtype=torch.float32,
            )
        if self.object_ref_conditioning:
            if "object_refs" not in inputs:
                raise ValueError("object_refs input is required by this checkpoint")
            model_kwargs["object_refs"] = torch.from_numpy(inputs["object_refs"]).to(
                self.device,
                dtype=torch.float32,
            )
        output = self.model(model_input, timesteps.flatten(), labels, **model_kwargs)
        if self.mask_prediction:
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(
                    "SmartBrush PTH denoiser must return image and mask logits"
                )
            output_map = {
                "output": output[0].detach().cpu().numpy(),
                "mask_logits": output[1].detach().cpu().numpy(),
            }
        else:
            output_map = {"output": output.detach().cpu().numpy()}
        return [output_map[name] for name in output_names]


def load_pth_session(model_in_file, train_config, device, use_ema):
    train_config_path = resolve_train_config(model_in_file, train_config)
    if not os.path.isfile(train_config_path):
        raise FileNotFoundError(f"train_config.json not found: {train_config_path}")

    opt = load_train_options(train_config_path, device)
    weights_path = resolve_weights_path(model_in_file, use_ema)
    model = build_model(opt, weights_path, device)
    return (
        PthDenoiserSession(
            model,
            device,
            mask_size_conditioning=bool(
                getattr(opt, "alg_b2b_mask_size_conditioning", False)
            ),
            mask_prediction=bool(getattr(opt, "alg_b2b_mask_prediction", False)),
            temporal_frame_step_conditioning=bool(
                getattr(opt, "alg_b2b_temporal_frame_step_conditioning", False)
            ),
            global_context_conditioning=b2b_global_context_enabled_from_opt(opt),
            object_ref_conditioning=bool(
                getattr(opt, "alg_b2b_object_ref_paths", None)
            ),
        ),
        weights_path,
        train_config_path,
    )


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
        "--fixed_temporal_init_noise",
        "--fixed-temporal-init-noise",
        action="store_true",
        help=(
            "Reuse one seeded two-frame noise field for every sliding window. "
            "Use with autoregressive reinjection to test temporal stability."
        ),
    )
    parser.add_argument(
        "--mask_precision_mode",
        "--mask-precision-mode",
        choices=sorted(onnx_runner.MASK_PRECISION_NAMES),
        help="SmartBrush coarse-mask precision mode. Defaults to bbox.",
    )
    parser.add_argument(
        "--mask_precision_severity",
        "--mask-precision-severity",
        type=float,
        help="SmartBrush coarse-mask severity in [0,1]. Defaults to 1.0.",
    )
    parser.add_argument(
        "--temporal_frame_step",
        type=float,
        help=(
            "Raw temporal frame stride for checkpoints trained with "
            "alg.b2b_temporal_frame_step_conditioning. Defaults to "
            "data.temporal_frame_step from train_config.json, then 1."
        ),
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        help="Override denoise step count. Defaults to first entry from alg.b2b_denoise_timesteps",
    )
    parser.add_argument(
        "--source_crop_size",
        "--source-crop-size",
        type=int,
        help=(
            "Inference-only square source crop size before resizing to the saved "
            "model resolution. Defaults to data.online_creation.crop_size_A."
        ),
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
        "--alg_b2b_object_ref_paths",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Override static object reference image paths for checkpoints trained "
            "with alg.b2b_object_ref_paths."
        ),
    )
    parser.add_argument(
        "--autoregressive_reinject_patch",
        "--autoregressive-reinject-patch",
        action="store_true",
        help=(
            "Feed the previously generated crop back as known context in the next "
            "sliding window, with zero projection and the predicted class mask "
            "as conditioning."
        ),
    )
    parser.add_argument(
        "--apply_predicted_mask",
        "--alg_b2b_apply_predicted_mask",
        action="store_true",
        help=(
            "Use the predicted mask for final compositing. By default the coarse "
            "input mask is used."
        ),
    )
    parser.add_argument(
        "--use_predicted_mask_during_denoising",
        "--alg_b2b_use_predicted_mask_during_denoising",
        action="store_true",
        help=(
            "Feed the predicted mask from each completed denoising interval into "
            "the following interval. Disabled by default."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = parse_device(args.device)

    session, weights_path, train_config_path = load_pth_session(
        args.model_in_file, args.train_config, device, args.use_ema
    )
    train_json, _ = onnx_runner.load_train_config(args.model_in_file, train_config_path)

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

    denoise_steps = onnx_runner.resolve_denoise_steps(train_json, args.denoise_steps)
    object_refs = onnx_runner.load_object_refs_for_inference(
        train_json, args.alg_b2b_object_ref_paths
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
        fixed_temporal_init_noise=args.fixed_temporal_init_noise,
        object_refs=object_refs,
        temporal_frame_step=args.temporal_frame_step,
        mask_precision_mode=args.mask_precision_mode,
        mask_precision_severity=args.mask_precision_severity,
        apply_predicted_mask=args.apply_predicted_mask,
        use_predicted_mask_during_denoising=(args.use_predicted_mask_during_denoising),
        source_crop_size=args.source_crop_size,
    )

    print(f"dataset_root : {dataset_root}")
    print(f"checkpoint   : {weights_path}")
    print(f"train_config : {train_config_path}")
    print(f"device       : {device}")
    print(f"train_shape  : {(train_frames, train_height, train_width)}")
    print(f"denoise_steps: {denoise_steps}")
    print(
        "source_crop_size: "
        f"{args.source_crop_size if args.source_crop_size is not None else 'saved crop_size_A'}"
    )
    print(
        "temporal_frame_step: "
        f"{onnx_runner.resolve_temporal_frame_step(train_json, args.temporal_frame_step)}"
    )
    print(
        "object_refs  : " f"{0 if object_refs is None else int(object_refs.shape[0])}"
    )
    precision_mode, precision_severity = onnx_runner.resolve_mask_precision(
        train_json, args.mask_precision_mode, args.mask_precision_severity
    )
    print(f"mask_precision: {(precision_mode, precision_severity)}")
    print(f"autoregressive_reinject_patch: {args.autoregressive_reinject_patch}")
    print(f"fixed_temporal_init_noise: {args.fixed_temporal_init_noise}")
    print(f"apply_predicted_mask: {args.apply_predicted_mask}")
    print(
        "use_predicted_mask_during_denoising: "
        f"{args.use_predicted_mask_during_denoising}"
    )
    print(f"written      : {len(frames_written)} frames")
    print(f"saved        : {args.output_dir}")


if __name__ == "__main__":
    main()
