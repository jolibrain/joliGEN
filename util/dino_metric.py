import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Metric(nn.Module):
    """Reference-based distance built from frozen DINOv2 image embeddings."""

    def __init__(self, model_name="dinov2_vitb14", image_size=224, model=None):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.model = model if model is not None else self._load_model(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _load_model(self, model_name):
        return torch.hub.load("facebookresearch/dinov2", model_name, force_reload=False)

    def _prepare_input(self, image):
        if image.ndim != 4:
            raise ValueError(
                f"DINOv2Metric expects 4D NCHW tensors, got shape {tuple(image.shape)}"
            )

        image = image.float()

        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] < 3:
            raise ValueError(
                "DINOv2Metric supports 1-channel grayscale or >=3-channel images"
            )
        elif image.shape[1] > 3:
            image = image[:, :3]

        if torch.amin(image) < 0:
            image = (image.clamp(-1, 1) + 1.0) / 2.0
        else:
            image = image.clamp(0, 1)

        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return (image - self.mean) / self.std

    def _extract_features(self, image):
        feats = self.model(image)

        if isinstance(feats, dict):
            if "x_norm_clstoken" in feats:
                feats = feats["x_norm_clstoken"]
            elif "cls_token" in feats:
                feats = feats["cls_token"]
            else:
                raise ValueError(
                    f"Unsupported DINOv2 output dictionary keys: {tuple(feats.keys())}"
                )

        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        if feats.ndim > 2:
            feats = feats.flatten(1)

        return F.normalize(feats, dim=1)

    @torch.inference_mode()
    def forward(self, prediction, target):
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                "DINOv2Metric expects prediction and target batches with matching size"
            )

        prediction = self._prepare_input(prediction)
        target = self._prepare_input(target)

        pred_feats = self._extract_features(prediction)
        target_feats = self._extract_features(target)

        return (1.0 - F.cosine_similarity(pred_feats, target_feats, dim=1)).mean()
