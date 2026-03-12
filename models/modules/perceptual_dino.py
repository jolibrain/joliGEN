import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

SUPPORTED_DINO_MODELS = {
    "dinov2_vitb14_reg": ("facebookresearch/dinov2", "dinov2_vitb14_reg"),
    "dinov2_vitl14_reg": ("facebookresearch/dinov2", "dinov2_vitl14_reg"),
    "dinov3_vitl16": ("facebookresearch/dinov3", "dinov3_vitl16"),
}


class DINOEncoder(nn.Module):
    def __init__(self, model="dinov2_vitb14_reg"):
        super().__init__()
        if model not in SUPPORTED_DINO_MODELS:
            raise ValueError(
                f"Unsupported DINO model {model}. "
                f"Choose from: {list(SUPPORTED_DINO_MODELS.keys())}"
            )

        repo_dir, model_name = SUPPORTED_DINO_MODELS[model]
        self.model = torch.hub.load(repo_dir, model_name).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, img):
        img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
        img = TF.normalize(
            img,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        return self.model.forward_features(img)["x_norm_patchtokens"]


class DINOPerceptualLoss(nn.Module):
    def __init__(
        self,
        model="dinov2_vitb14_reg",
        resize_resolution=224,
    ):
        super().__init__()
        if resize_resolution <= 0:
            raise ValueError("resize_resolution must be a positive integer")

        self.resize_resolution = resize_resolution
        self.encoder = DINOEncoder(model=model).eval()
        self.encoder.requires_grad_(False)

    def forward(self, target, pred):
        if target.ndim != 4 or pred.ndim != 4:
            raise ValueError(
                "DINOPerceptualLoss expects 4D tensors shaped [B, C, H, W]"
            )
        if target.shape[1] != 3 or pred.shape[1] != 3:
            raise ValueError(
                "DINOPerceptualLoss requires 3-channel perceptual inputs"
            )
        if target.shape[0] != pred.shape[0]:
            raise ValueError("target and pred must have the same batch size")

        if target.shape[-2:] != (self.resize_resolution, self.resize_resolution):
            target = F.interpolate(
                target,
                size=(self.resize_resolution, self.resize_resolution),
                mode="bilinear",
                align_corners=False,
            )
            pred = F.interpolate(
                pred,
                size=(self.resize_resolution, self.resize_resolution),
                mode="bilinear",
                align_corners=False,
            )

        pred_features = self.encoder(pred.float())
        with torch.no_grad():
            target_features = self.encoder(target.detach().float())

        cos_sim = F.cosine_similarity(
            pred_features.float(),
            target_features.float(),
            dim=-1,
        )
        return (1.0 - cos_sim).mean()
