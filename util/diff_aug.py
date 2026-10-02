import math
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class DiffAugment:
    SUPPORTED_POLICIES = {
        "camera_color",
        "color",
        "detail",
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

    CAMERA_RGB_GAIN = (0.85, 1.18)
    CAMERA_TEMPERATURE = (-0.20, 0.20)
    CAMERA_TINT = (-0.10, 0.10)
    CAMERA_EXPOSURE_EV = (-1.0, 1.0)
    CAMERA_CHANNEL_OFFSET = (-20.0 / 255.0, 20.0 / 255.0)
    CAMERA_BLACK_POINT = (-0.04, 0.04)
    CAMERA_WHITE_POINT = (0.90, 1.10)
    CAMERA_GAMMA = (0.70, 1.40)
    CAMERA_SATURATION = (0.80, 1.20)
    CAMERA_TONE = (-0.50, 0.50)
    CAMERA_MATRIX_OFF_DIAGONAL = (-0.04, 0.04)
    CAMERA_GRADIENT = (-0.15, 0.15)
    CAMERA_VIGNETTE = (0.0, 0.35)
    CAMERA_VIGNETTE_CENTER = (-0.20, 0.20)

    DETAIL_MODE_WEIGHTS = {
        "soften": 0.20,
        "unsharp": 0.45,
        "clarity": 0.20,
        "phone": 0.15,
    }

    def __init__(
        self,
        policy="",
        p=0.0,
        camera_color_strength=1.0,
        detail_strength=1.0,
    ):
        self.p = p
        self.camera_color_strength = self._validate_strength(
            "camera_color_strength", camera_color_strength
        )
        self.detail_strength = self._validate_strength(
            "detail_strength", detail_strength
        )
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
            image_tensors, _, _ = self._apply_policy_to_sample([x], [], [None])
            return image_tensors[0]

        image_tensors, _ = self.apply_synchronized(image_tensors=[x], mask_tensors=[])
        return image_tensors[0]

    def apply_synchronized(
        self,
        image_tensors=None,
        mask_tensors=None,
        image_exclusion_masks=None,
        excluded_policies=None,
    ):
        """Apply one sampled augmentation plan to related tensors.

        Image and mask tensors may use different spatial resolutions. Spatial
        parameters are sampled against the first tensor, then scaled to preserve
        the same normalized transform for every other tensor. An optional
        exclusion mask can be associated with each image; non-zero exclusion
        pixels are transformed geometrically and restored to normalized black
        after augmentation.
        """
        image_tensors = [] if image_tensors is None else list(image_tensors)
        mask_tensors = [] if mask_tensors is None else list(mask_tensors)
        excluded_policies = set(excluded_policies or ())
        if image_exclusion_masks is None:
            image_exclusion_masks = [None] * len(image_tensors)
        else:
            image_exclusion_masks = list(image_exclusion_masks)
        if len(image_exclusion_masks) != len(image_tensors):
            raise ValueError(
                "image_exclusion_masks must have one entry per image tensor"
            )
        self._validate_image_exclusion_masks(image_tensors, image_exclusion_masks)

        tensors = [
            tensor
            for tensor in image_tensors + mask_tensors + image_exclusion_masks
            if tensor is not None
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
        aug_exclusion_masks = []
        for tensor in image_exclusion_masks:
            aug_exclusion_masks.append(None if tensor is None else tensor.clone())

        for batch_idx in range(batch_size):
            image_sample = [
                None if tensor is None else tensor[batch_idx] for tensor in aug_images
            ]
            mask_sample = [
                None if tensor is None else tensor[batch_idx] for tensor in aug_masks
            ]
            exclusion_sample = [
                None if tensor is None else tensor[batch_idx]
                for tensor in aug_exclusion_masks
            ]
            image_sample, mask_sample, exclusion_sample = self._apply_policy_to_sample(
                image_sample,
                mask_sample,
                exclusion_sample,
                excluded_policies,
            )
            for tensor_idx, sample in enumerate(image_sample):
                if sample is not None:
                    aug_images[tensor_idx][batch_idx] = sample
            for tensor_idx, sample in enumerate(mask_sample):
                if sample is not None:
                    aug_masks[tensor_idx][batch_idx] = sample
            for tensor_idx, sample in enumerate(exclusion_sample):
                if sample is not None:
                    aug_exclusion_masks[tensor_idx][batch_idx] = sample

        return aug_images, aug_masks

    def _validate_image_exclusion_masks(self, image_tensors, exclusion_masks):
        for image, exclusion_mask in zip(image_tensors, exclusion_masks):
            if exclusion_mask is None:
                continue
            if image is None:
                raise ValueError("An exclusion mask cannot target a missing image")
            compatible_shape = (
                exclusion_mask.ndim == image.ndim
                and exclusion_mask.shape[:-3] == image.shape[:-3]
                and exclusion_mask.shape[-3] in (1, image.shape[-3])
                and exclusion_mask.shape[-2:] == image.shape[-2:]
            )
            if not compatible_shape:
                raise RuntimeError(
                    "Each image exclusion mask must match its image tensor except "
                    "for an optional singleton channel dimension"
                )

    def _validate_strength(self, name, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
        return value

    def sample_camera_color_plan(self):
        """Return a sampled plan, or None when the policy is disabled/skipped."""
        if (
            "camera_color" not in self.policy_names
            or self.p <= 0
            or random.uniform(0, 1) >= self.p
        ):
            return None
        return self._sample_camera_color_params()

    def apply_camera_color_plan(self, tensor, params, reference_size=None):
        """Apply a previously sampled camera plan to one normalized tensor."""
        if params is None:
            return tensor
        if reference_size is None:
            reference_size = tensor.shape[-2:]
        return self._apply_camera_color(tensor, params, reference_size)

    def _apply_policy_to_sample(
        self,
        image_tensors,
        mask_tensors,
        image_exclusion_masks,
        excluded_policies=None,
    ):
        tensors = [
            tensor
            for tensor in image_tensors + mask_tensors + image_exclusion_masks
            if tensor is not None
        ]
        if not tensors:
            return image_tensors, mask_tensors, image_exclusion_masks

        height, width = tensors[0].shape[-2:]
        reference_size = (height, width)
        excluded_policies = set(excluded_policies or ())
        for policy_name in self.policy_names:
            if policy_name in excluded_policies:
                continue
            if random.uniform(0, 1) >= self.p:
                continue
            if policy_name == "camera_color":
                if not any(tensor is not None for tensor in image_tensors):
                    continue
                params = self._sample_camera_color_params()
                image_tensors = [
                    (
                        None
                        if tensor is None
                        else self._apply_camera_color(
                            tensor,
                            params,
                            reference_size,
                        )
                    )
                    for tensor in image_tensors
                ]
            elif policy_name == "color":
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
            elif policy_name == "detail":
                if not any(tensor is not None for tensor in image_tensors):
                    continue
                params = self._sample_detail_params()
                image_tensors = [
                    (
                        None
                        if tensor is None
                        else self._apply_detail(tensor, params, reference_size)
                    )
                    for tensor in image_tensors
                ]
            elif policy_name == "randaffine":
                params = transforms.RandomAffine.get_params(
                    self.AFFINE_DEGREES,
                    self.AFFINE_TRANSLATE,
                    self.AFFINE_SCALE,
                    self.AFFINE_SHEAR,
                    [width, height],
                )
                image_tensors = [
                    (
                        None
                        if tensor is None
                        else self._apply_affine(
                            tensor,
                            params,
                            InterpolationMode.BILINEAR,
                            reference_size,
                        )
                    )
                    for tensor in image_tensors
                ]
                mask_tensors = [
                    (
                        None
                        if tensor is None
                        else self._apply_affine(
                            tensor,
                            params,
                            InterpolationMode.NEAREST,
                            reference_size,
                        )
                    )
                    for tensor in mask_tensors
                ]
                image_exclusion_masks = [
                    (
                        None
                        if tensor is None
                        else self._apply_affine(
                            tensor,
                            params,
                            InterpolationMode.NEAREST,
                            reference_size,
                            fill_value=1.0,
                        )
                    )
                    for tensor in image_exclusion_masks
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
                            tensor,
                            params,
                            InterpolationMode.BILINEAR,
                            reference_size,
                        )
                    )
                    for tensor in image_tensors
                ]
                mask_tensors = [
                    (
                        None
                        if tensor is None
                        else self._apply_perspective(
                            tensor,
                            params,
                            InterpolationMode.NEAREST,
                            reference_size,
                        )
                    )
                    for tensor in mask_tensors
                ]
                image_exclusion_masks = [
                    (
                        None
                        if tensor is None
                        else self._apply_perspective(
                            tensor,
                            params,
                            InterpolationMode.NEAREST,
                            reference_size,
                            fill_value=1.0,
                        )
                    )
                    for tensor in image_exclusion_masks
                ]

        image_tensors = [
            None if tensor is None else tensor.clamp(-1.0, 1.0)
            for tensor in image_tensors
        ]
        image_tensors = [
            self._restore_excluded_pixels(tensor, exclusion_mask)
            for tensor, exclusion_mask in zip(
                image_tensors,
                image_exclusion_masks,
            )
        ]
        return image_tensors, mask_tensors, image_exclusion_masks

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

    def _apply_camera_color(self, tensor, params, reference_size):
        if self.camera_color_strength == 0.0:
            return tensor
        orig_dtype = tensor.dtype
        tensor = tensor.float()
        tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
        original_shape = tensor.shape
        frames = tensor.reshape(
            -1, original_shape[-3], original_shape[-2], original_shape[-1]
        )

        spatial_gain = self._camera_spatial_gain(
            frames,
            params,
            reference_size,
        )
        frames = frames * spatial_gain
        frames = frames * params["exposure"]

        if frames.shape[1] == 3:
            gains = frames.new_tensor(params["rgb_gains"]).view(1, 3, 1, 1)
            frames = frames * gains
            matrix = frames.new_tensor(params["color_matrix"])
            frames = torch.einsum("ij,bjhw->bihw", matrix, frames)
            offsets = frames.new_tensor(params["channel_offsets"]).view(1, 3, 1, 1)
            frames = frames + offsets

        frames = (frames - params["black_point"]) / (
            params["white_point"] - params["black_point"]
        )
        frames = frames.clamp(0.0, 1.0).pow(params["gamma"])

        if frames.shape[1] == 3:
            luminance_weights = frames.new_tensor([0.2126, 0.7152, 0.0722]).view(
                1, 3, 1, 1
            )
            luminance = (frames * luminance_weights).sum(dim=1, keepdim=True)
            frames = luminance + params["saturation"] * (frames - luminance)

        tone = params["tone"]
        frames = frames + tone * frames * (1.0 - frames) * (2.0 * frames - 1.0)
        tensor = frames.reshape(original_shape).clamp(0.0, 1.0)
        return (tensor * 2.0 - 1.0).to(orig_dtype)

    def _camera_spatial_gain(self, frames, params, reference_size):
        del reference_size  # Coordinates are normalized independently of resolution.
        height, width = frames.shape[-2:]
        if height <= 1:
            y_coords = frames.new_zeros((height,))
        else:
            y_coords = torch.linspace(-1.0, 1.0, height, device=frames.device)
        if width <= 1:
            x_coords = frames.new_zeros((width,))
        else:
            x_coords = torch.linspace(-1.0, 1.0, width, device=frames.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        gradient = 1.0 + params["gradient_x"] * xx + params["gradient_y"] * yy
        center_x, center_y = params["vignette_center"]
        radius_squared = ((xx - center_x).square() + (yy - center_y).square()) / 2.0
        vignette = 1.0 - params["vignette"] * radius_squared
        return (gradient * vignette).clamp(0.5, 1.5).view(1, 1, height, width)

    def _apply_detail(self, tensor, params, reference_size):
        if params["strength"] == 0.0:
            return tensor
        orig_dtype = tensor.dtype
        tensor = tensor.float()
        tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
        original_shape = tensor.shape
        frames = tensor.reshape(
            -1, original_shape[-3], original_shape[-2], original_shape[-1]
        )
        strength = params["strength"]
        mode = params["mode"]

        if mode == "soften":
            blurred = self._gaussian_blur_scaled(
                frames, params["soften_sigma"], reference_size
            )
            frames = frames + strength * (blurred - frames)
        elif mode == "unsharp":
            blurred = self._gaussian_blur_scaled(
                frames, params["unsharp_sigma"], reference_size
            )
            frames = frames + strength * params["unsharp_amount"] * (frames - blurred)
        elif mode == "clarity":
            blurred = self._gaussian_blur_scaled(
                frames, params["clarity_sigma"], reference_size
            )
            frames = frames + strength * params["clarity_amount"] * (frames - blurred)
        elif mode == "phone":
            denoised = self._gaussian_blur_scaled(
                frames, params["phone_denoise_sigma"], reference_size
            )
            frames = frames + strength * (denoised - frames)
            blurred = self._gaussian_blur_scaled(
                frames, params["phone_unsharp_sigma"], reference_size
            )
            frames = frames + strength * params["phone_unsharp_amount"] * (
                frames - blurred
            )
            local_mean = self._gaussian_blur_scaled(
                frames, params["phone_clarity_sigma"], reference_size
            )
            frames = frames + strength * params["phone_clarity_amount"] * (
                frames - local_mean
            )
        else:
            raise RuntimeError(f"Unsupported detail mode: {mode}")

        tensor = frames.reshape(original_shape).clamp(0.0, 1.0)
        return (tensor * 2.0 - 1.0).to(orig_dtype)

    def _gaussian_blur_scaled(self, frames, sigma, reference_size):
        reference_height, reference_width = reference_size
        height, width = frames.shape[-2:]
        scale = min(
            height / max(1, reference_height),
            width / max(1, reference_width),
        )
        sigma = max(0.01, float(sigma) * scale)
        max_kernel = min(height, width)
        if max_kernel % 2 == 0:
            max_kernel -= 1
        if max_kernel < 3:
            return frames
        kernel_size = min(2 * int(math.ceil(3.0 * sigma)) + 1, max_kernel)
        kernel_size = max(3, kernel_size)
        return TF.gaussian_blur(
            frames,
            kernel_size=[kernel_size, kernel_size],
            sigma=[sigma, sigma],
        )

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

    def _sample_camera_color_params(self):
        strength = self.camera_color_strength

        def enabled(probability):
            return random.uniform(0, 1) < probability

        def additive(bounds, active=True):
            if not active:
                return 0.0
            return random.uniform(*bounds) * strength

        def multiplicative(bounds, active=True):
            if not active or strength == 0:
                return 1.0
            value = random.uniform(*bounds)
            return math.exp(math.log(value) * strength)

        white_balance = enabled(0.80)
        if white_balance:
            residual_gains = [multiplicative(self.CAMERA_RGB_GAIN) for _ in range(3)]
            temperature = additive(self.CAMERA_TEMPERATURE)
            tint = additive(self.CAMERA_TINT)
            temperature_gains = (1.0 + temperature, 1.0, 1.0 - temperature)
            tint_gains = (1.0, 1.0 + tint, 1.0)
            rgb_gains = [
                residual * temperature_gain * tint_gain
                for residual, temperature_gain, tint_gain in zip(
                    residual_gains,
                    temperature_gains,
                    tint_gains,
                )
            ]
        else:
            rgb_gains = [1.0, 1.0, 1.0]

        exposure_ev = additive(self.CAMERA_EXPOSURE_EV, enabled(0.70))
        offsets_active = enabled(0.50)
        channel_offsets = [
            additive(self.CAMERA_CHANNEL_OFFSET, offsets_active) for _ in range(3)
        ]

        levels_active = enabled(0.50)
        black_point = additive(self.CAMERA_BLACK_POINT, levels_active)
        white_point = 1.0 + additive(
            (
                self.CAMERA_WHITE_POINT[0] - 1.0,
                self.CAMERA_WHITE_POINT[1] - 1.0,
            ),
            levels_active,
        )

        matrix = [[0.0] * 3 for _ in range(3)]
        matrix_active = enabled(0.25)
        for row in range(3):
            off_diagonal_sum = 0.0
            for column in range(3):
                if row == column:
                    continue
                value = additive(self.CAMERA_MATRIX_OFF_DIAGONAL, matrix_active)
                matrix[row][column] = value
                off_diagonal_sum += value
            matrix[row][row] = 1.0 - off_diagonal_sum

        vignette_active = enabled(0.35)
        vignette = additive(self.CAMERA_VIGNETTE, vignette_active)
        vignette_center = (
            additive(self.CAMERA_VIGNETTE_CENTER, vignette_active),
            additive(self.CAMERA_VIGNETTE_CENTER, vignette_active),
        )
        gradient_active = enabled(0.35)

        return {
            "rgb_gains": rgb_gains,
            "exposure": 2.0**exposure_ev,
            "channel_offsets": channel_offsets,
            "black_point": black_point,
            "white_point": white_point,
            "gamma": multiplicative(self.CAMERA_GAMMA, enabled(0.50)),
            "saturation": multiplicative(self.CAMERA_SATURATION, enabled(0.50)),
            "tone": additive(self.CAMERA_TONE, enabled(0.50)),
            "color_matrix": matrix,
            "gradient_x": additive(self.CAMERA_GRADIENT, gradient_active),
            "gradient_y": additive(self.CAMERA_GRADIENT, gradient_active),
            "vignette": vignette,
            "vignette_center": vignette_center,
        }

    def _sample_detail_params(self):
        modes = list(self.DETAIL_MODE_WEIGHTS)
        mode = random.choices(
            modes,
            weights=[self.DETAIL_MODE_WEIGHTS[name] for name in modes],
            k=1,
        )[0]
        return {
            "mode": mode,
            "strength": self.detail_strength,
            "soften_sigma": random.uniform(0.15, 1.0),
            "unsharp_sigma": random.uniform(0.30, 1.20),
            "unsharp_amount": random.uniform(0.25, 2.0),
            "clarity_sigma": random.uniform(1.50, 4.0),
            "clarity_amount": random.uniform(0.10, 0.75),
            "phone_denoise_sigma": random.uniform(0.15, 0.60),
            "phone_unsharp_sigma": random.uniform(0.30, 1.0),
            "phone_unsharp_amount": random.uniform(0.50, 1.50),
            "phone_clarity_sigma": random.uniform(1.50, 3.0),
            "phone_clarity_amount": random.uniform(0.0, 0.40),
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

    def _apply_affine(
        self,
        tensor,
        params,
        interpolation,
        reference_size=None,
        fill_value=0.0,
    ):
        if reference_size is not None:
            params = self._scale_affine_params(
                params,
                reference_size,
                tensor.shape[-2:],
            )
        angle, translate, scale, shear = params
        return TF.affine(
            tensor,
            angle=angle,
            translate=list(translate),
            scale=scale,
            shear=list(shear),
            interpolation=interpolation,
            fill=self._make_fill(tensor, fill_value),
        )

    def _apply_perspective(
        self,
        tensor,
        params,
        interpolation,
        reference_size=None,
        fill_value=0.0,
    ):
        if reference_size is not None:
            params = self._scale_perspective_params(
                params,
                reference_size,
                tensor.shape[-2:],
            )
        startpoints, endpoints = params
        return TF.perspective(
            tensor,
            startpoints=startpoints,
            endpoints=endpoints,
            interpolation=interpolation,
            fill=self._make_fill(tensor, fill_value),
        )

    def _scale_affine_params(self, params, reference_size, target_size):
        angle, translate, scale, shear = params
        reference_height, reference_width = reference_size
        target_height, target_width = target_size
        scaled_translate = (
            translate[0] * target_width / reference_width,
            translate[1] * target_height / reference_height,
        )
        return angle, scaled_translate, scale, shear

    def _scale_perspective_params(self, params, reference_size, target_size):
        reference_height, reference_width = reference_size
        target_height, target_width = target_size
        x_scale = self._perspective_axis_scale(reference_width, target_width)
        y_scale = self._perspective_axis_scale(reference_height, target_height)

        def scale_points(points):
            return [[point[0] * x_scale, point[1] * y_scale] for point in points]

        startpoints, endpoints = params
        return scale_points(startpoints), scale_points(endpoints)

    def _perspective_axis_scale(self, reference_size, target_size):
        if reference_size <= 1:
            return 1.0
        return (target_size - 1) / (reference_size - 1)

    def _restore_excluded_pixels(self, tensor, exclusion_mask):
        if tensor is None or exclusion_mask is None:
            return tensor
        return tensor.masked_fill(exclusion_mask > 0.5, -1.0)

    def _sample_factor(self, amount):
        return random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)

    def _make_fill(self, tensor, value=0.0):
        channels = tensor.shape[-3]
        return [value] * channels
