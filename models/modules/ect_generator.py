import math
import sys
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from torch import nn, Tensor
from einops.layers.torch import Rearrange

# from models.modules.diffusion_utils import gamma_embedding


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


def output_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the model's output.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the model's output.
    """
    return (sigma_data * (sigma - sigma_min)) / (sigma_data**2 + sigma**2) ** 0.5


def output_scaling_train(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the model's output.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the model's output.
    """
    return (sigma_data * sigma) / (sigma_data**2 + sigma**2) ** 0.5


def skip_scaling(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the residual connection.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the residual connection.
    """
    return sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)


def skip_scaling_train(
    sigma: Tensor, sigma_data: float = 0.5, sigma_min: float = 0.002
) -> Tensor:
    """Computes the scaling value for the residual connection.

    Parameters
    ----------
    sigma : Tensor
        Current standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise from the karras schedule.

    Returns
    -------
    Tensor
        Scaling value for the residual connection.
    """
    return sigma_data**2 / (sigma**2 + sigma_data**2)


class NoiseLevelEmbedding(nn.Module):
    def __init__(self, channels: int, scale: float = 0.02) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)
        #
        #        self.projection = nn.Sequential(
        #            nn.Linear(channels, 4 * channels),
        #            nn.SiLU(),
        #            nn.Linear(4 * channels, channels),
        #            Rearrange("b c -> b c () ()"),
        #        )
        ##
        self.projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            Rearrange("b c -> b c () ()"),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x[:, None] * self.W[None, :] * 2 * torch.pi
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)

        return self.projection(h)


class ECTGenerator(nn.Module):
    def __init__(
        self,
        ect_model,
        sampling_method,
        image_size,
        G_ngf,
        double_ticks=1000,
    ):
        super().__init__()

        self.ect_model = ect_model
        self.sampling_method = sampling_method
        self.image_size = image_size

        self.sigma_min = 0.002
        self.sigma_max = float("inf")  # 80
        self.sigma_data = 0.5

        ##ECT
        self.P_mean = -1.1
        self.P_std = 2.0
        self.q = 256.0
        self.stage = 0
        self.k = 8.0
        self.b = 1.0
        self.c = 0.0  # 0.000001
        self.stage = 0
        self.current_t = 0
        self.double_ticks = double_ticks
        self.ratio = 1 - 1 / self.q ** (self.stage + 1)

        self.cond_embed_dim = self.ect_model.cond_embed_dim
        self.ect_cond_embed = NoiseLevelEmbedding(self.cond_embed_dim)

        self.current_t = 0
        self.double_ticks = double_ticks

        self.t_to_r = self.t_to_r_sigmoid

    def t_to_r_sigmoid(self, k, b, q, t, stage):

        adj = 1 + k * torch.sigmoid(-b * t)
        decay = 1 / q ** (stage + 1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def ect_forward(self, x, sigma, sigma_data, sigma_min, x_cond=None):
        if self.training:
            c_skip = skip_scaling_train(sigma, sigma_data, sigma_min)
            c_out = output_scaling_train(sigma, sigma_data, sigma_min)
        else:  # phase=="test"
            c_skip = skip_scaling(sigma, sigma_data, sigma_min)
            c_out = output_scaling(sigma, sigma_data, sigma_min)

        # Pad dimensions as broadcasting will not work
        c_skip = pad_dims_like(c_skip, x)
        c_out = pad_dims_like(c_out, x)

        embed_noise_level = self.embed_sigmas(sigma)
        if not x_cond is None:
            if len(x.shape) != 5:
                x_with_cond = torch.cat([x_cond, x], dim=1)
            else:
                x_with_cond = torch.cat([x_cond, x], dim=2)
        else:
            if len(x.shape) != 5:
                x_with_cond = torch.cat([x, x], dim=1)
            else:
                x_with_cond = torch.cat([x, x], dim=2)

        return c_skip * x + c_out * self.ect_model(
            x_with_cond, embed_noise_level
        )  # , **kwargs)

    def forward(
        self,
        x,
        total_training_steps=50000,
        mask=None,
        x_cond=None,
    ):
        rnd_normal = torch.randn(x.shape[0], device=x.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()
        r = self.t_to_r(self.k, self.b, self.q, t, self.stage)

        eps = torch.randn_like(x)
        eps_t = eps * t
        eps_r = eps * r

        if mask is not None:
            mask = torch.clamp(mask, min=0.0, max=1.0)

        t_noisy_x = x + pad_dims_like(eps_t, x)
        if mask is not None:
            t_noisy_x = t_noisy_x * mask + (1 - mask) * x

        D_yt = self.ect_forward(
            t_noisy_x,
            t,
            self.sigma_data,
            self.sigma_min,
            x_cond,
        )
        with torch.no_grad():
            r_noisy_x = x + pad_dims_like(eps_r, x)
            if mask is not None:
                r_noisy_x = r_noisy_x * mask + (1 - mask) * x

            D_yr = self.ect_forward(
                r_noisy_x,
                r,
                self.sigma_data,
                self.sigma_min,
                x_cond,
            )

        bs = x.size(dim=0)
        self.current_t += bs

        print(
            f"[ECTGenerator] Stage  is {self.stage} | ratio={self.ratio:.6f} | current_t={self.current_t} | r is {r.view(-1)} and t is {t.view(-1)}"
        )

        return (
            D_yt,
            D_yr,
            t_noisy_x,
            r_noisy_x,
            t,
            r,
        )

    # External tick-based stage update (new)
    # ------------------------
    def update_stage(self, cur_tick):  ### <<< NEW FUNCTION
        new_stage = cur_tick // self.double_ticks
        if new_stage > self.stage:
            self.stage = new_stage
            self.ratio = 1 - 1 / self.q ** (self.stage + 1)
            print(f"[ECT] Stage updated → {self.stage} | ratio={self.ratio:.6f}")

    def restoration(self, y, y_cond, sigmas, mask, clip_denoised=True):
        if mask is not None:
            mask = torch.clamp(
                mask, min=0.0, max=1.0
            )  # removes class information from mask
            y = y * (1 - mask)

        # Sample at the end of the schedule
        x = y + sigmas[0] * torch.randn_like(y)

        if mask is not None:
            x = x * mask + (1 - mask) * y

        sigma = torch.full((x.shape[0],), sigmas[0], dtype=x.dtype, device=x.device)
        x = self.ect_forward(
            x,
            sigma,
            self.sigma_data,
            self.sigma_min,
            y_cond,
        )
        if clip_denoised:
            x = x.clamp(min=-1.0, max=1.0)

        if mask is not None:
            x = x * mask + (1 - mask) * y

        for sigma in sigmas[1:]:

            sigma = torch.full((x.shape[0],), sigma, dtype=x.dtype, device=x.device)
            x = x + pad_dims_like(
                (sigma**2 - self.sigma_min**2) ** 0.5, x
            ) * torch.randn_like(x)

            if mask is not None:
                x = x * mask + (1 - mask) * y

            x = self.ect_forward(
                x,
                sigma,
                self.sigma_data,
                self.sigma_min,
                y_cond,
            )

            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)
            if mask is not None:
                x = x * mask + (1 - mask) * y

        return x

    def embed_sigmas(self, sigmas):
        emb = self.ect_cond_embed(sigmas).squeeze(dim=[2, 3])
        return emb
