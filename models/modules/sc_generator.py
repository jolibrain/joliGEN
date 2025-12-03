import math
import sys
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from torch import nn, Tensor
from einops.layers.torch import Rearrange

# from models.modules.diffusion_utils import gamma_embedding

import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)

        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:  # pad if dim is odd
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (N,) Tensor of timesteps
        Returns:
            (N, hidden_size) embedded representation
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def pad_dims_like(x: Tensor, other: Tensor) -> Tensor:
    """Pad dimensions of tensor `x` to match the shape of tensor `other`.

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    other : Tensor
        Tensor whose shape will be used as reference for padding.

    Returns
    -------
    Tensor
        Padded tensor with the same shape as other.
    """
    ndim = other.ndim - x.ndim
    return x.view(*x.shape, *((1,) * ndim))


class SCGenerator(nn.Module):
    """
    Shortcut Consistency Generator (CM-style)
    Handles both target generation and model forward prediction internally.
    """

    def __init__(
        self,
        sc_model,
        sampling_method,
        image_size,
        G_ngf,
        num_timesteps=128,
        bootstrap_ratio=0.125,
        force_dt=-1.0,
    ):
        super().__init__()
        self.sc_model = sc_model
        self.num_timesteps = num_timesteps
        self.bootstrap_ratio = bootstrap_ratio
        self.force_dt = force_dt
        self.sampling_method = sampling_method
        self.image_size = image_size

        # Time and delta-time embeddings
        self.cond_embed_dim = self.sc_model.cond_embed_dim
        self.t_embedder = TimestepEmbedder(self.cond_embed_dim)
        self.dt_embedder = TimestepEmbedder(self.cond_embed_dim)  # 1152
        self.current_t = 1

    # Like cm_forward
    def sc_forward(self, x_t, t, dt, x_cond=None):
        """Run the neural network once."""

        t = self.t_embedder(t)  # (N, D)
        dt = self.dt_embedder(dt)
        embed_t_dt = t + dt

        if not x_cond is None:
            if len(x.shape) != 5:
                x_with_cond = torch.cat([x_cond, x_t], dim=1)
            else:
                x_with_cond = torch.cat([x_cond, x_t], dim=2)
        else:
            x_with_cond = x_t

        return self.sc_model(x_with_cond, embed_t_dt)

    # Like forward() in CMGenerator
    def forward(self, x, mask=None, x_cond=None):
        """
        Main forward pass for training.
        Generates targets + runs shortcut model internally.
        """
        FORCE_T = -1
        FORCE_DT = -1
        device = x.device
        batch_size = x.shape[0]  # x shape is [bs, frame_seq, 3, w, h] within [-1, 1]
        info = {}
        denoise_timesteps = self.num_timesteps
        log2_sections = int(np.log2(denoise_timesteps))

        # === Sample t and dt_base ===
        # 1. create step sizes dt
        dt_base = torch.randint(
            low=0, high=log2_sections, size=(batch_size,), device=device
        ).float()
        dt = 1.0 / (2.0**dt_base)

        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2  # [0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 0.5]

        # 2. sample timesteps t
        dt_sections = 2**dt_base

        t = torch.cat(
            [
                torch.randint(low=0, high=int(val.item()), size=(1,)).float()
                for val in dt_sections
            ]
        ).to(device)

        t = t / dt_sections
        t_full = t[(...,) + (None,) * (x.ndim - t.ndim)]

        # === Base data ===
        if mask is not None:
            mask = torch.clamp(mask, min=0.0, max=1.0)

        x_1 = x
        x_0 = torch.randn_like(x_1)

        x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1

        if mask is not None:
            x_t = x_t * mask + (1 - mask) * x_1

        # === Mode decision ===
        use_bootstrap = torch.rand(1).item() < self.bootstrap_ratio

        if use_bootstrap:
            # -------- Bootstrap Mode --------
            with torch.no_grad():
                v_b1 = self.sc_forward(x_t, t, dt_base_bootstrap, x_cond)

            t2 = t + dt_bootstrap
            dt_bootstrap_full = dt_bootstrap[
                (...,) + (None,) * (x.ndim - dt_bootstrap.ndim)
            ]
            x_t2 = x_t + dt_bootstrap_full * v_b1
            x_t2 = torch.clamp(x_t2, -1.5, 1.5)  # check if 2 or 4 ??

            with torch.no_grad():
                v_b2 = self.sc_forward(x_t2, t2, dt_base_bootstrap, x_cond)

            v_target = (v_b1 + v_b2) / 2

            v_target = torch.clip(v_target, -1.5, 1.5)  # check if 2 or 4??

            v_pred = self.sc_forward(x_t, t, dt_base, x_cond)
            info["mode"] = "bootstrap"

        else:
            # -------- Flow Matching Mode --------
            v_target = x_1 - (1 - 1e-5) * x_0
            v_pred = self.sc_forward(x_t, t, dt_base, x_cond)
            info["mode"] = "flow"

        return v_pred, v_target, t, dt_base, x_t, info

    def restoration(self, y, y_cond, denoise_timesteps, mask=None, clip_denoised=True):

        B = y.shape[0]
        device = y.device  # ??maybe no necessary

        if mask is not None:
            mask = torch.clamp(
                mask, min=0.0, max=1.0
            )  # removes class information from mask
            y = y * (1 - mask)

        x = y + torch.randn_like(y)

        if mask is not None:
            x = x * mask + (1 - mask) * y
        delta_t = 1.0 / denoise_timesteps  # i.e. step size
        for ti in range(denoise_timesteps):
            t = ti / denoise_timesteps

            t_vector = torch.full((B,), t).to(device)
            dt_base = torch.ones_like(t_vector).to(device) * math.log2(
                denoise_timesteps
            )

            with torch.no_grad():
                v = self.sc_forward(x, t_vector, dt_base, y_cond)

            x = x + v * delta_t

            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)
            if mask is not None:
                x = x * mask + (1 - mask) * y

        return x
