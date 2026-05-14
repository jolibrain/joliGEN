import math
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class DiffAugment:
    SUPPORTED_POLICIES = {
        "color",
        "wild",
        "color+wild",
        "randaffine",
        "randperspective",
    }
    COLOR_BRIGHTNESS = 0.2
    COLOR_CONTRAST = 0.2
    COLOR_SATURATION = 0.2
    COLOR_HUE = 0.02
    AFFINE_DEGREES = [-30.0, 30.0]
    AFFINE_TRANSLATE = (0.05, 0.05)
    AFFINE_SCALE = (0.8, 1.0)
    AFFINE_SHEAR = [-15.0, 15.0]
    PERSPECTIVE_DISTORTION = 0.5
    WILD_INTERPOLATION_MODES = ("area", "bilinear", "bicubic")
    WILD_STAGE_CONFIGS = (
        {
            "blur_p": 0.60,
            "blur_sigma": (0.15, 1.0),
            "resize_p": 0.80,
            "resize_scale": (0.60, 1.0),
            "noise_p": 0.80,
            "noise_std": (1.0 / 255.0, 0.05),
            "compression_p": 0.25,
            "compression_levels": (96, 255),
        },
        {
            "blur_p": 0.30,
            "blur_sigma": (0.10, 0.60),
            "resize_p": 0.50,
            "resize_scale": (0.75, 1.0),
            "noise_p": 0.50,
            "noise_std": (0.5 / 255.0, 0.05),
            "compression_p": 0.15,
            "compression_levels": (128, 255),
        },
    )

    def __init__(self, policy="", p=0.0):
        self.p = p
        self.policy_names = [name.strip() for name in policy.split(",") if name.strip()]
        invalid = [
            name for name in self.policy_names if name not in self.SUPPORTED_POLICIES
        ]
        if invalid:
            raise ValueError(
                f"Unsupported DiffAug policy {invalid}. "
                f"Allowed values: {sorted(self.SUPPORTED_POLICIES)}"
            )

    def __call__(self, x):
        if x.ndim == 3:
            image_tensors, _ = self._apply_policy_to_sample([x], [])
            return image_tensors[0]

        image_tensors, _ = self.apply_synchronized(image_tensors=[x], mask_tensors=[])
        return image_tensors[0]

    def apply_synchronized(self, image_tensors=None, mask_tensors=None):
        image_tensors = [] if image_tensors is None else list(image_tensors)
        mask_tensors = [] if mask_tensors is None else list(mask_tensors)
        tensors = [
            tensor for tensor in image_tensors + mask_tensors if tensor is not None
        ]
        if not tensors or self.p <= 0 or not self.policy_names:
            return image_tensors, mask_tensors

        batch_size = tensors[0].shape[0]
        for tensor in tensors[1:]:
            if tensor.shape[0] != batch_size:
                raise RuntimeError("All tensors must share the same batch dimension")

        aug_images = []
        for tensor in image_tensors:
            aug_images.append(None if tensor is None else tensor.clone())
        aug_masks = []
        for tensor in mask_tensors:
            aug_masks.append(None if tensor is None else tensor.clone())

        for batch_idx in range(batch_size):
            image_sample = [
                None if tensor is None else tensor[batch_idx] for tensor in aug_images
            ]
            mask_sample = [
                None if tensor is None else tensor[batch_idx] for tensor in aug_masks
            ]
            image_sample, mask_sample = self._apply_policy_to_sample(
                image_sample, mask_sample
            )
            for tensor_idx, sample in enumerate(image_sample):
                if sample is not None:
                    aug_images[tensor_idx][batch_idx] = sample
            for tensor_idx, sample in enumerate(mask_sample):
                if sample is not None:
                    aug_masks[tensor_idx][batch_idx] = sample

        return aug_images, aug_masks

    def _apply_policy_to_sample(self, image_tensors, mask_tensors):
        tensors = [
            tensor for tensor in image_tensors + mask_tensors if tensor is not None
        ]
        if not tensors:
            return image_tensors, mask_tensors

        height, width = tensors[0].shape[-2:]
        for policy_name in self.policy_names:
            if random.uniform(0, 1) >= self.p:
                continue
            if policy_name == "color":
                if not any(tensor is not None for tensor in image_tensors):
                    continue
                params = self._sample_color_params(image_tensors)
                image_tensors = [
                    None if tensor is None else self._apply_color(tensor, params)
                    for tensor in image_tensors
                ]
            elif policy_name == "wild":
                if not any(tensor is not None for tensor in image_tensors):
                    continue
                params = self._sample_wild_params()
                image_tensors = [
                    None if tensor is None else self._apply_wild(tensor, params)
                    for tensor in image_tensors
                ]
            elif policy_name == "color+wild":
                if not any(tensor is not None for tensor in image_tensors):
                    continue
                color_params = self._sample_color_params(image_tensors)
                image_tensors = [
                    None if tensor is None else self._apply_color(tensor, color_params)
                    for tensor in image_tensors
                ]
                wild_params = self._sample_wild_params()
                image_tensors = [
                    None if tensor is None else self._apply_wild(tensor, wild_params)
                    for tensor in image_tensors
                ]
            elif policy_name == "randaffine":
                params = transforms.RandomAffine.get_params(
                    self.AFFINE_DEGREES,
                    self.AFFINE_TRANSLATE,
                    self.AFFINE_SCALE,
                    self.AFFINE_SHEAR,
                    [height, width],
                )
                image_tensors = [
                    (
                        None
                        if tensor is None
                        else self._apply_affine(
                            tensor, params, InterpolationMode.BILINEAR
                        )
                    )
                    for tensor in image_tensors
                ]
                mask_tensors = [
                    (
                        None
                        if tensor is None
                        else self._apply_affine(
                            tensor, params, InterpolationMode.NEAREST
                        )
                    )
                    for tensor in mask_tensors
                ]
            elif policy_name == "randperspective":
                params = transforms.RandomPerspective.get_params(
                    width, height, self.PERSPECTIVE_DISTORTION
                )
                image_tensors = [
                    (
                        None
                        if tensor is None
                        else self._apply_perspective(
                            tensor, params, InterpolationMode.BILINEAR
                        )
                    )
                    for tensor in image_tensors
                ]
                mask_tensors = [
                    (
                        None
                        if tensor is None
                        else self._apply_perspective(
                            tensor, params, InterpolationMode.NEAREST
                        )
                    )
                    for tensor in mask_tensors
                ]

        image_tensors = [
            None if tensor is None else tensor.clamp(-1.0, 1.0)
            for tensor in image_tensors
        ]
        return image_tensors, mask_tensors

    def _apply_color(self, tensor, params):
        orig_dtype = tensor.dtype
        tensor = tensor.float()
        tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)

        brightness = params["brightness"]
        contrast = params["contrast"]
        saturation = params["saturation"]
        hue = params["hue"]
        ops = list(params["ops"])

        for op_name in ops:
            if op_name == "brightness":
                tensor = TF.adjust_brightness(tensor, brightness)
            elif op_name == "contrast":
                tensor = TF.adjust_contrast(tensor, contrast)
            elif op_name == "saturation":
                tensor = TF.adjust_saturation(tensor, saturation)
            elif op_name == "hue":
                tensor = TF.adjust_hue(tensor, hue)

        tensor = tensor.clamp(0.0, 1.0)
        tensor = tensor * 2.0 - 1.0
        return tensor.to(orig_dtype)

    def _apply_wild(self, tensor, params):
        orig_dtype = tensor.dtype
        tensor = tensor.float()
        tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
        original_shape = tensor.shape
        frames = tensor.reshape(
            -1, original_shape[-3], original_shape[-2], original_shape[-1]
        )

        for stage in params["stages"]:
            if stage["blur"]:
                frames = TF.gaussian_blur(
                    frames,
                    kernel_size=stage["blur_kernel_size"],
                    sigma=[stage["blur_sigma"], stage["blur_sigma"]],
                )
            if stage["resize"]:
                frames = self._apply_wild_resize(frames, stage)
            if stage["noise"]:
                frames = self._apply_wild_noise(frames, stage, params)
            if stage["compression"]:
                levels = stage["compression_levels"]
                frames = torch.round(frames.clamp(0.0, 1.0) * levels) / levels

        tensor = frames.reshape(original_shape).clamp(0.0, 1.0)
        tensor = tensor * 2.0 - 1.0
        return tensor.to(orig_dtype)

    def _apply_wild_resize(self, frames, stage):
        height, width = frames.shape[-2:]
        scaled_height = max(1, int(round(height * stage["resize_scale"])))
        scaled_width = max(1, int(round(width * stage["resize_scale"])))
        if scaled_height == height and scaled_width == width:
            return frames

        mode = stage["resize_mode"]
        down_kwargs = {"mode": mode}
        up_kwargs = {"mode": mode}
        if mode in {"bilinear", "bicubic"}:
            down_kwargs["align_corners"] = False
            up_kwargs["align_corners"] = False
        frames = F.interpolate(
            frames, size=(scaled_height, scaled_width), **down_kwargs
        )
        frames = F.interpolate(frames, size=(height, width), **up_kwargs)
        return frames

    def _apply_wild_noise(self, frames, stage, params):
        noise_key = (tuple(frames.shape), frames.device, frames.dtype, stage["index"])
        noise_tensors = params["noise_tensors"]
        if noise_key not in noise_tensors:
            if stage["noise_type"] == "gaussian":
                noise = torch.randn_like(frames) * stage["noise_sigma"]
                noise_tensors[noise_key] = noise
            else:
                quantized = torch.clamp((frames * 255.0).round(), 0, 255) / 255.0
                vals = self._poisson_value_counts(quantized)
                poisson = torch.poisson(quantized * vals) / vals
                noise_tensors[noise_key] = (poisson - quantized) * stage["noise_sigma"]
        return (frames + noise_tensors[noise_key]).clamp(0.0, 1.0)

    def _poisson_value_counts(self, frames):
        vals_list = []
        for frame in frames:
            unique_count = max(1, torch.unique(frame).numel())
            vals_list.append(2 ** math.ceil(math.log2(unique_count)))
        return frames.new_tensor(vals_list).view(-1, 1, 1, 1)

    def _sample_color_params(self, image_tensors):
        sample = next(tensor for tensor in image_tensors if tensor is not None)
        ops = ["brightness", "contrast"]
        if sample.shape[-3] == 3:
            ops += ["saturation", "hue"]
        random.shuffle(ops)
        return {
            "brightness": self._sample_factor(self.COLOR_BRIGHTNESS),
            "contrast": self._sample_factor(self.COLOR_CONTRAST),
            "saturation": self._sample_factor(self.COLOR_SATURATION),
            "hue": random.uniform(-self.COLOR_HUE, self.COLOR_HUE),
            "ops": ops,
        }

    def _sample_wild_params(self):
        stages = []
        for index, config in enumerate(self.WILD_STAGE_CONFIGS):
            blur = random.uniform(0, 1) < config["blur_p"]
            resize = random.uniform(0, 1) < config["resize_p"]
            noise = random.uniform(0, 1) < config["noise_p"]
            compression = random.uniform(0, 1) < config["compression_p"]
            noise_type = "gaussian" if random.uniform(0, 1) < 0.7 else "poisson"
            stages.append(
                {
                    "index": index,
                    "blur": blur,
                    "blur_sigma": random.uniform(*config["blur_sigma"]),
                    "blur_kernel_size": 3,
                    "resize": resize,
                    "resize_scale": random.uniform(*config["resize_scale"]),
                    "resize_mode": random.choice(self.WILD_INTERPOLATION_MODES),
                    "noise": noise,
                    "noise_type": noise_type,
                    "noise_sigma": random.uniform(*config["noise_std"]),
                    "compression": compression,
                    "compression_levels": random.randint(*config["compression_levels"]),
                }
            )
        return {"stages": stages, "noise_tensors": {}}

    def _apply_affine(self, tensor, params, interpolation):
        angle, translate, scale, shear = params
        return TF.affine(
            tensor,
            angle=angle,
            translate=list(translate),
            scale=scale,
            shear=list(shear),
            interpolation=interpolation,
            fill=self._make_fill(tensor),
        )

    def _apply_perspective(self, tensor, params, interpolation):
        startpoints, endpoints = params
        return TF.perspective(
            tensor,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=interpolation,
            fill=self._make_fill(tensor),
        )

    def _sample_factor(self, amount):
        return random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)

    def _make_fill(self, tensor):
        channels = tensor.shape[-3]
        return [0.0] * channels
