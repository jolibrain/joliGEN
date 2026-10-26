"""Clean-image residual refinement, with optional within-window temporal mixing.

Spatial design: MFlow ConvRefiner (arXiv:2610.23854v1, Fig. 7).
Temporal branches and the 64-channel default are JoliGEN adaptations.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .discriminators import NLayerDiscriminator


def frame_timesteps(t, batch, frames):
    """Accept a timestep per sample or per frame, as JiTViD does."""
    t = t.reshape(batch, -1)
    return t.expand(batch, frames).reshape(-1)


class RefinerBlock(nn.Module):
    def __init__(self, width, temporal):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.time = nn.Linear(width, width)
        self.norm2 = nn.GroupNorm(32, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.temporal = temporal
        if temporal:
            self.temporal_norm = nn.GroupNorm(32, width)
            self.temporal_conv = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0))
            nn.init.zeros_(self.temporal_conv.weight)
            nn.init.zeros_(self.temporal_conv.bias)

    def forward(self, h, time, batch, frames):
        residual = self.conv1(F.silu(self.norm1(h)))
        residual = residual + self.time(F.silu(time))[:, :, None, None]
        h = h + self.conv2(F.silu(self.norm2(residual)))
        if self.temporal:
            # Normalize each frame independently; only the Conv3d mixes frames.
            residual = F.silu(self.temporal_norm(h))
            residual = residual.reshape(batch, frames, *h.shape[1:]).transpose(1, 2)
            residual = self.temporal_conv(residual).transpose(1, 2).reshape_as(h)
            h = h + residual
        return h


class ConvRefiner(nn.Module):
    def __init__(self, channels=3, width=64, blocks=4, temporal=False):
        super().__init__()
        if width < 32 or width % 32:
            raise ValueError("ConvRefiner width must be a positive multiple of 32")
        self.register_buffer(
            "time_frequencies",
            torch.exp(
                -math.log(10000) * torch.arange(width // 2).float() / (width // 2)
            ),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.input_conv = nn.Conv2d(channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            [RefinerBlock(width, temporal) for _ in range(blocks)]
        )
        self.output_norm = nn.GroupNorm(32, width)
        self.output_conv = nn.Conv2d(width, channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, prediction, timesteps):
        video = prediction.ndim == 5
        batch = prediction.shape[0]
        frames = prediction.shape[1] if video else 1
        x = prediction.flatten(0, 1) if video else prediction
        t = frame_timesteps(timesteps, batch, frames).to(x.dtype)
        phase = t[:, None] * self.time_frequencies[None].to(x.dtype) * 1000
        time = self.time_embed(torch.cat([phase.cos(), phase.sin()], dim=-1))
        h = self.input_conv(x)
        for block in self.blocks:
            h = block(h, time, batch, frames)
        result = x + self.output_conv(F.silu(self.output_norm(h)))
        return result.reshape_as(prediction)


class RefinerDiscriminator(nn.Module):
    """Framewise t-conditioned PatchGAN; no temporal discriminator is implied."""

    def __init__(self, channels=3):
        super().__init__()
        self.patchgan = NLayerDiscriminator(
            channels + 1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d
        )

    def forward(self, images, timesteps):
        time = timesteps.reshape(-1, 1, 1, 1).expand(-1, 1, *images.shape[-2:])
        return self.patchgan(torch.cat([images, time.to(images.dtype)], dim=1))

    def patch_weights(self, mask):
        # Coverage over each logit's actual convolutional receptive field.
        coverage = mask.float()
        for layer in self.patchgan.model:
            if isinstance(layer, nn.Conv2d):
                coverage = F.avg_pool2d(
                    coverage,
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    count_include_pad=False,
                )
        return coverage


def weighted_patch_mean(value, weight):
    """Empty masks retain a gradient path and contribute zero on every rank."""
    return (value * weight).sum() / weight.sum().clamp_min(1)


def load_refiner_generator_state(generator, state, fresh=False):
    """Only a fresh finetune may omit the newly initialized refiner."""
    if isinstance(state, dict) and "_ema" in state:
        state = state["_ema"]
    expected = generator.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    allowed = {key for key in missing if fresh and key.startswith("convrefiner.")}
    if missing - allowed or unexpected:
        raise RuntimeError(
            "Incompatible ConvRefiner checkpoint: "
            f"missing={sorted(missing - allowed)}, unexpected={sorted(unexpected)}"
        )
    generator._allow_fresh_refiner_load = fresh
    try:
        generator.load_state_dict(state, strict=not fresh)
    finally:
        generator._allow_fresh_refiner_load = False
