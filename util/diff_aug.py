import random

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class DiffAugment:
    SUPPORTED_POLICIES = {"color", "randaffine", "randperspective"}
    COLOR_BRIGHTNESS = 0.2
    COLOR_CONTRAST = 0.2
    COLOR_SATURATION = 0.2
    COLOR_HUE = 0.02
    AFFINE_DEGREES = [-30.0, 30.0]
    AFFINE_TRANSLATE = (0.05, 0.05)
    AFFINE_SCALE = (0.8, 1.0)
    AFFINE_SHEAR = [-15.0, 15.0]
    PERSPECTIVE_DISTORTION = 0.5

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

        self.transform_list = []
        if p > 0:
            for name in self.policy_names:
                self.transform_list.append(AUGMENT_FNS[name])

    def __call__(self, x):
        for transform in self.transform_list:
            if random.uniform(0, 1) < self.p:
                x = transform(x)
        return x

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
                params = self._sample_color_params(image_tensors)
                image_tensors = [
                    None if tensor is None else self._apply_color(tensor, params)
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
        mask_tensors = [
            None if tensor is None else tensor.clamp(0.0, 1.0)
            for tensor in mask_tensors
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


AUGMENT_FNS = {
    "color": transforms.ColorJitter(),
    "randaffine": transforms.RandomAffine(
        [-30, 30],
        (0.05, 0.05),
        (0.8, 1),
        (-15, 15),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    "randperspective": transforms.RandomPerspective(
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
}
