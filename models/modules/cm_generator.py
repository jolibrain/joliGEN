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


def improved_timesteps_schedule(
    current_training_step: int,
    total_training_steps: int,
    initial_timesteps: int = 10,
    final_timesteps: int = 1280,
) -> int:
    """Implements the improved timestep discretization schedule.

    Parameters
    ----------
    current_training_step : int
        Current step in the training loop.
    total_training_steps : int
        Total number of steps the model will be trained for.
    initial_timesteps : int, default=2
        Timesteps at the start of training.
    final_timesteps : int, default=150
        Timesteps at the end of training.

    Returns
    -------
    int
        Number of timesteps at the current point in training.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    total_training_steps_prime = math.floor(
        total_training_steps
        / (math.log2(math.floor(final_timesteps / initial_timesteps)) + 1)
    )
    num_timesteps = initial_timesteps * math.pow(
        2, math.floor(current_training_step / total_training_steps_prime)
    )
    num_timesteps = min(num_timesteps, final_timesteps) + 1

    return num_timesteps


def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
) -> Tensor:
    """Implements the karras schedule that controls the standard deviation of
    noise added.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    sigma_min : float, default=0.002
        Minimum standard deviation.
    sigma_max : float, default=80.0
        Maximum standard deviation
    rho : float, default=7.0
        Schedule hyper-parameter.
    device : torch.device, default=None
        Device to generate the schedule/sigmas/boundaries/ts on.

    Returns
    -------
    Tensor
        Generated schedule/sigmas/boundaries/ts.
    """
    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (sigma_max**rho_inv - sigma_min**rho_inv)
    sigmas = sigmas**rho

    return sigmas


def lognormal_timestep_distribution(
    num_samples: int,
    sigmas: Tensor,
    mean: float = -1.1,
    std: float = 2.0,
) -> Tensor:
    """Draws timesteps from a lognormal distribution.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw.
    sigmas : Tensor
        Standard deviations of the noise.
    mean : float, default=-1.1
        Mean of the lognormal distribution.
    std : float, default=2.0
        Standard deviation of the lognormal distribution.

    Returns
    -------
    Tensor
        Timesteps drawn from the lognormal distribution.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
        (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
    )
    pdf = pdf / pdf.sum()

    timesteps = torch.multinomial(pdf, num_samples, replacement=True)

    return timesteps


def improved_loss_weighting(sigmas: Tensor) -> Tensor:
    """Computes the weighting for the consistency loss.

    Parameters
    ----------
    sigmas : Tensor
        Standard deviations of the noise.

    Returns
    -------
    Tensor
        Weighting for the consistency loss.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    return 1 / (sigmas[1:] - sigmas[:-1])


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
    def __init__(self, channels: int, opt, scale: float = 0.02) -> None:

        super().__init__()

        self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)
        if getattr(opt, "alg_diffusion_ddpm_cm_ft", False):
            self.projection = nn.Sequential(
                nn.Linear(channels, channels),
                nn.SiLU(),
                nn.Linear(channels, channels),
                Rearrange("b c -> b c () ()"),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(channels, 4 * channels),
                nn.SiLU(),
                nn.Linear(4 * channels, channels),
                Rearrange("b c -> b c () ()"),
            )

    def forward(self, x: Tensor) -> Tensor:
        h = x[:, None] * self.W[None, :] * 2 * torch.pi
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)

        return self.projection(h)


class CMGenerator(nn.Module):
    def __init__(
        self,
        cm_model,
        sampling_method,
        image_size,
        G_ngf,
        opt=None,
    ):
        super().__init__()

        self.cm_model = cm_model
        self.sampling_method = sampling_method
        self.image_size = image_size
        self.opt = opt

        self.sigma_min = 0.002
        self.sigma_max = 80.0
        self.sigma_data = 0.5

        # improved consistency training
        self.rho = 7.0
        self.initial_timesteps = 10
        self.final_timesteps = 1280
        self.lognormal_mean = -1.1
        self.lognormal_std = 2.0

        self.cond_embed_dim = self.cm_model.cond_embed_dim
        self.cm_cond_embed = NoiseLevelEmbedding(self.cond_embed_dim, self.opt)

        self.current_t = 2  # default value, set from cm_model upon resume

        ##ECT
        self.P_mean = -1.1
        self.P_std = 2.0
        self.q = 2.0
        self.stage = 0
        self.k = 8.0
        self.b = 1.0
        self.stage = 0
        self.current_t = 0
        self.double_ticks = 1000
        self.ratio = 1 - 1 / self.q ** (self.stage + 1)
        self.t_to_r = self.t_to_r_sigmoid

    def t_to_r_sigmoid(self, k, b, q, t, stage):
        adj = 1 + k * torch.sigmoid(-b * t)
        decay = 1 / q ** (stage + 1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def update_stage(self, cur_tick):
        new_stage = cur_tick // self.double_ticks
        if new_stage > self.stage:
            self.stage = new_stage
            self.ratio = 1 - 1 / self.q ** (self.stage + 1)
            print(f"[ECT] Stage updated → {self.stage} | ratio={self.ratio:.6f}")

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
            x_with_cond = x
        #            if len(x.shape) != 5 and not self.opt.alg_diffusion_ddpm_cm_ft:
        #                x_with_cond = torch.cat([x, x], dim=1)
        #            elif self.opt.alg_ddpm_ft_mode == "ect":
        #                x_with_cond = torch.cat([x, x], dim=1)
        #            else:
        #                x_with_cond = torch.cat([x, x], dim=2)
        #
        return c_skip * x + c_out * self.cm_model(
            x_with_cond, embed_noise_level
        )  # , **kwargs)

    def cm_forward(self, x, sigma, sigma_data, sigma_min, x_cond=None):
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
            elif self.opt.G_netG == "unet_vid" and self.opt.alg_diffusion_ddpm_cm_ft:
                x_with_cond = torch.cat([x, x], dim=2)
            else:
                x_with_cond = x

        return c_skip * x + c_out * self.cm_model(
            x_with_cond, embed_noise_level
        )  # , **kwargs)

    def forward(
        self,
        x,
        total_training_steps=50000,
        mask=None,
        x_cond=None,
    ):

        if self.opt.alg_ddpm_ft_mode == "ect":
            rnd_normal = torch.randn(x.shape[0], device=x.device)
            t = (rnd_normal * self.P_std + self.P_mean).exp()
            r = self.t_to_r(self.k, self.b, self.q, t, self.stage)
            eps = torch.randn_like(x)
            if mask is not None:
                mask = torch.clamp(mask, min=0.0, max=1.0)

            t_noisy_x = x + pad_dims_like(t, x) * eps
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
                r_noisy_x = x + pad_dims_like(r, x) * eps
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

        num_timesteps = improved_timesteps_schedule(
            self.current_t,
            total_training_steps,
            self.initial_timesteps,
            self.final_timesteps,
        )
        sigmas = karras_schedule(
            num_timesteps, self.sigma_min, self.sigma_max, self.rho, x.device
        )
        noise = torch.randn_like(x)
        timesteps = lognormal_timestep_distribution(
            x.shape[0], sigmas, self.lognormal_mean, self.lognormal_std
        )

        current_sigmas = sigmas[timesteps]
        next_sigmas = sigmas[timesteps + 1]

        if mask is not None:
            mask = torch.clamp(mask, min=0.0, max=1.0)

        next_noisy_x = x + pad_dims_like(next_sigmas, x) * noise
        if mask is not None:
            next_noisy_x = next_noisy_x * mask + (1 - mask) * x

        next_x = self.cm_forward(
            next_noisy_x,
            next_sigmas,
            self.sigma_data,
            self.sigma_min,
            x_cond,
        )

        with torch.no_grad():
            current_noisy_x = x + pad_dims_like(current_sigmas, x) * noise
            if mask is not None:
                current_noisy_x = current_noisy_x * mask + (1 - mask) * x

            current_x = self.cm_forward(
                current_noisy_x,
                current_sigmas,
                self.sigma_data,
                self.sigma_min,
                x_cond,
            )

        loss_weights = pad_dims_like(improved_loss_weighting(sigmas)[timesteps], next_x)

        bs = x.size(dim=0)
        self.current_t += bs

        return (
            next_x,
            current_x,
            num_timesteps,
            sigmas,
            loss_weights,
            next_noisy_x,
            current_noisy_x,
        )

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
        x = self.cm_forward(
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

            x = self.cm_forward(
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
        emb = self.cm_cond_embed(sigmas).squeeze(dim=[2, 3])
        return emb
