import argparse
import json
import math
import os
import sys

import torch


JG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(JG_DIR)

from models import diffusion_networks
from options.train_options import TrainOptions
from models.modules import b2b_generator as b2b_generator_module
from models.modules.vit import vit_vid as vit_vid_module


def patch_vit_vid_attention():
    def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(
            query.size(0), 1, L, S, dtype=torch.float32, device=query.device
        )

        with torch.cuda.amp.autocast(enabled=False):
            attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            output = attn_weight @ value.float()
        return output

    vit_vid_module.scaled_dot_product_attention = scaled_dot_product_attention


def patch_b2b_tracing_compatibility():
    @torch.no_grad()
    def _forward_sample_restoration(self, x, t, y_cond, mask, labels, y_known):
        x_in = self._project_known_pixels(x, y_known, mask)
        if labels is None:
            labels = torch.zeros(x_in.shape[0], dtype=torch.long, device=x_in.device)
        if y_cond is None:
            model_input = x_in
        elif len(x_in.shape) != 5:
            model_input = torch.cat([y_cond, x_in], dim=1)
        else:
            model_input = torch.cat([y_cond, x_in], dim=2)

        x_cond = self.b2b_model(model_input, t.flatten(), labels)
        x_cond = self._match_prediction_channels(x_cond, x_in)
        x_cond = self._project_known_pixels(x_cond, y_known, mask)

        den = 1.0 - t
        if not self.disable_inference_clipping:
            den = den.clamp_min(self.t_eps)
        v_cond = (x_cond - x_in) / den

        if math.isclose(self.cfg_scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
            return v_cond

        low, high = self.cfg_interval
        interval_mask = t < high
        if low != 0:
            interval_mask = interval_mask & (t > low)

        num_classes = int(self.b2b_model.num_classes)
        x_uncond = self.b2b_model(
            model_input, t.flatten(), torch.full_like(labels, num_classes)
        )
        x_uncond = self._match_prediction_channels(x_uncond, x_in)
        x_uncond = self._project_known_pixels(x_uncond, y_known, mask)
        v_uncond = (x_uncond - x_in) / den

        cfg_scale_interval = torch.where(
            interval_mask,
            torch.full_like(t, self.cfg_scale),
            torch.ones_like(t),
        )

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    b2b_generator_module.B2BGenerator._forward_sample_restoration = (
        _forward_sample_restoration
    )


def load_train_options(train_config_path, device):
    with open(train_config_path, "r") as f:
        train_json = json.load(f)

    if device.type == "cuda":
        train_json["gpu_ids"] = str(device.index if device.index is not None else 0)
    else:
        train_json["gpu_ids"] = "-1"

    opt = TrainOptions().parse_json(train_json, save_config=False, set_device=False)
    opt.jg_dir = JG_DIR
    if opt.model_type in ["cm", "cm_gan", "sc", "b2b"]:
        opt.alg_palette_sampling_method = "ddim"
        opt.alg_diffusion_cond_embed_dim = 256
        if (
            opt.alg_diffusion_cond_image_creation == "computed_sketch"
            and opt.G_netG in ["unet_vid", "vit_vid"]
        ):
            opt.alg_diffusion_cond_embed = opt.alg_diffusion_cond_image_creation
    return opt


def resolve_steps(opt, requested_steps):
    if requested_steps is not None:
        return int(requested_steps)

    steps = getattr(opt, "alg_b2b_denoise_timesteps", 50)
    if isinstance(steps, int):
        return steps
    if not steps:
        raise ValueError("alg_b2b_denoise_timesteps is empty in train_config.json")
    return int(max(steps))


def resolve_output_path(model_in_file, model_out_file, export_mode, denoise_steps):
    if model_out_file:
        return model_out_file

    stem, _ext = os.path.splitext(model_in_file)
    if export_mode == "restoration":
        return f"{stem}_restoration_steps{denoise_steps}.onnx"
    return f"{stem}_denoiser.onnx"


def resolve_weights_path(model_in_file, use_ema):
    if not use_ema:
        return model_in_file

    if not model_in_file.endswith(".pth"):
        raise ValueError("--use_ema expects a .pth checkpoint path")

    ema_path = model_in_file[:-4] + "_ema.pth"
    if not os.path.isfile(ema_path):
        raise FileNotFoundError(f"EMA checkpoint not found: {ema_path}")
    return ema_path


def build_model(opt, weights_path, device):
    if opt.model_type != "b2b":
        raise ValueError(
            f"This exporter only supports b2b checkpoints, got model_type={opt.model_type!r}"
        )

    patch_vit_vid_attention()
    patch_b2b_tracing_compatibility()

    model = diffusion_networks.define_G(opt=opt, **vars(opt))
    weights = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        print(f"warning: missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"warning: unexpected keys while loading checkpoint: {len(unexpected)}")
    model.float()
    model.eval()
    model.to(device)
    return model


class B2BRestorationWrapperNoCond(torch.nn.Module):
    def __init__(self, model, denoise_steps):
        super().__init__()
        self.model = model
        self.denoise_steps = int(denoise_steps)

    def forward(self, y, mask, init_noise, labels):
        return self.model.restoration(
            y=y,
            y_cond=None,
            denoise_timesteps=self.denoise_steps,
            mask=mask,
            labels=labels,
            init_noise=init_noise,
        )


class B2BRestorationWrapperWithCond(torch.nn.Module):
    def __init__(self, model, denoise_steps):
        super().__init__()
        self.model = model
        self.denoise_steps = int(denoise_steps)

    def forward(self, y, cond_image, mask, init_noise, labels):
        return self.model.restoration(
            y=y,
            y_cond=cond_image,
            denoise_timesteps=self.denoise_steps,
            mask=mask,
            labels=labels,
            init_noise=init_noise,
        )


class B2BDenoiserWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.b2b_model

    def forward(self, model_input, timesteps, labels):
        return self.model(model_input, timesteps, labels)


def build_export_wrapper(model, opt, export_mode, denoise_steps):
    cond_creation = getattr(opt, "alg_diffusion_cond_image_creation", "y_t")
    uses_cond = cond_creation != "y_t"

    if export_mode == "denoiser":
        return B2BDenoiserWrapper(model), uses_cond

    if uses_cond:
        return B2BRestorationWrapperWithCond(model, denoise_steps), uses_cond
    return B2BRestorationWrapperNoCond(model, denoise_steps), uses_cond


def make_dummy_inputs(opt, device, export_mode, uses_cond, batch_size, num_frames):
    height = int(opt.data_crop_size)
    width = int(opt.data_crop_size)
    output_nc = int(opt.model_output_nc)

    y = torch.zeros(batch_size, num_frames, output_nc, height, width, device=device)
    mask = torch.ones(batch_size, num_frames, 1, height, width, device=device)
    init_noise = torch.zeros_like(y)
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    if export_mode == "denoiser":
        in_channels = int(model_input_channels_for_b2b(opt))
        model_input = torch.zeros(
            batch_size, num_frames, in_channels, height, width, device=device
        )
        timesteps = torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)
        return (model_input, timesteps, labels)

    if uses_cond:
        cond_image = torch.zeros_like(y)
        return (y, cond_image, mask, init_noise, labels)
    return (y, mask, init_noise, labels)


def model_input_channels_for_b2b(opt):
    input_nc = int(opt.model_input_nc)
    if getattr(opt, "alg_b2b_mask_as_channel", False):
        input_nc += 1
    if "mask" in getattr(opt, "alg_diffusion_cond_embed", ""):
        input_nc += int(getattr(opt, "alg_diffusion_cond_embed_dim", 0))
    if getattr(opt, "alg_diffusion_cond_image_creation", "y_t") != "y_t":
        input_nc += int(opt.model_output_nc)
    return input_nc


def export_to_onnx(
    wrapper,
    dummy_inputs,
    model_out_file,
    export_mode,
    uses_cond,
    opset_version,
    dynamic_batch_frames,
):
    if export_mode == "denoiser":
        input_names = ["model_input", "timesteps", "labels"]
    elif uses_cond:
        input_names = ["y", "cond_image", "mask", "init_noise", "labels"]
    else:
        input_names = ["y", "mask", "init_noise", "labels"]

    output_names = ["output"]
    dynamic_axes = None
    if dynamic_batch_frames:
        dynamic_axes = {"output": {0: "batch", 1: "frames"}}
        for name in input_names:
            if name == "labels" or name == "timesteps":
                dynamic_axes[name] = {0: "batch"}
            else:
                dynamic_axes[name] = {0: "batch", 1: "frames"}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            model_out_file,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )


def parse_device(device_name):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA export requested but torch.cuda.is_available() is False"
        )
    return device


def check_onnx(model_out_file):
    try:
        import onnx
    except ImportError:
        print("warning: onnx is not installed, skipped ONNX validation")
        return

    model = onnx.load(model_out_file)
    onnx.checker.check_model(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_in_file",
        required=True,
        help="Path to the joliGEN b2b checkpoint (.pth file)",
    )
    parser.add_argument(
        "--train_config",
        help="Path to train_config.json. Defaults to the checkpoint directory train_config.json",
    )
    parser.add_argument(
        "--model_out_file",
        help="Path to the exported ONNX model. Defaults next to the checkpoint",
    )
    parser.add_argument(
        "--export_mode",
        default="restoration",
        choices=["restoration", "denoiser"],
        help="Export the full B2B restoration graph or only the raw vit/denoiser backbone",
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        help="Fixed denoise steps for restoration export. Defaults to max(alg_b2b_denoise_timesteps) from train_config.json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Dummy batch size used for export",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        help="Dummy frame count used for export. Defaults to data_temporal_number_frames from train_config.json",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Export the sibling *_ema.pth checkpoint instead of the provided .pth file",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=18,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dynamic_batch_frames",
        action="store_true",
        help="Export batch and frame axes as dynamic",
    )
    parser.add_argument(
        "--skip_onnx_check",
        action="store_true",
        help="Skip onnx.checker validation after export",
    )

    args = parser.parse_args()

    device = parse_device(args.device)

    train_config = args.train_config
    if not train_config:
        train_config = os.path.join(
            os.path.dirname(os.path.abspath(args.model_in_file)), "train_config.json"
        )
    if not os.path.isfile(train_config):
        raise FileNotFoundError(f"train_config.json not found: {train_config}")

    opt = load_train_options(train_config, device)
    denoise_steps = resolve_steps(opt, args.denoise_steps)
    weights_path = resolve_weights_path(args.model_in_file, args.use_ema)
    model_out_file = resolve_output_path(
        args.model_in_file, args.model_out_file, args.export_mode, denoise_steps
    )

    if getattr(opt, "alg_b2b_mask_as_channel", False):
        raise ValueError(
            "This exporter does not support alg_b2b_mask_as_channel=True yet"
        )

    model = build_model(opt, weights_path, device)
    wrapper, uses_cond = build_export_wrapper(
        model, opt, args.export_mode, denoise_steps
    )
    wrapper.eval()
    wrapper.to(device)

    num_frames = args.num_frames
    if num_frames is None:
        num_frames = int(getattr(opt, "data_temporal_number_frames", 1))

    dummy_inputs = make_dummy_inputs(
        opt=opt,
        device=device,
        export_mode=args.export_mode,
        uses_cond=uses_cond,
        batch_size=args.batch_size,
        num_frames=num_frames,
    )

    print(f"checkpoint     : {weights_path}")
    print(f"train_config   : {train_config}")
    print(f"export_mode    : {args.export_mode}")
    print(f"device         : {device}")
    print(f"denoise_steps  : {denoise_steps}")
    print(f"batch_size     : {args.batch_size}")
    print(f"num_frames     : {num_frames}")
    print(f"output         : {model_out_file}")

    export_to_onnx(
        wrapper=wrapper,
        dummy_inputs=dummy_inputs,
        model_out_file=model_out_file,
        export_mode=args.export_mode,
        uses_cond=uses_cond,
        opset_version=args.opset_version,
        dynamic_batch_frames=args.dynamic_batch_frames,
    )

    if not args.skip_onnx_check:
        check_onnx(model_out_file)

    print("export complete")


if __name__ == "__main__":
    main()
