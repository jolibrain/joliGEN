import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from torch import nn

from models.modules.unet_generator_attn.unet_generator_attn import UNet
from models.modules.diffusion_utils import (
    set_new_noise_schedule,
    predict_start_from_noise,
    q_posterior,
)

from models.modules.resnet_architecture.resnet_generator_diff import (
    ResnetGenerator_attn_diff,
)


class DiffusionGenerator(nn.Module):
    def __init__(
        self,
        denoise_fn,
    ):

        super().__init__()

        self.denoise_fn = denoise_fn

        # Init noise schedule
        set_new_noise_schedule(model=self.denoise_fn, phase="train")
        set_new_noise_schedule(model=self.denoise_fn, phase="test")

    def restoration(
        self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, x_a=None
    ):
        phase = "test"

        b, *_ = y_cond.shape

        assert (
            self.denoise_fn.num_timesteps_test > sample_num
        ), "num_timesteps must greater than sample_num"
        sample_inter = self.denoise_fn.num_timesteps_test // sample_num

        y_t = self.default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(
            reversed(range(0, self.denoise_fn.num_timesteps_test)),
            desc="sampling loop time step",
            total=self.denoise_fn.num_timesteps_test,
        ):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond, phase=phase, x_a=x_a)
            if mask is not None:
                temp_mask = torch.clamp(mask, min=0.0, max=1.0)
                y_t = y_0 * (1.0 - temp_mask) + temp_mask * y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)

            y_t, ret_arr = y_t.detach(), ret_arr.detach()

        return y_t, ret_arr

    def exists(self, x):
        return x is not None

    def default(self, val, d):
        if self.exists(val):
            return val
        return d() if isfunction(d) else d

    def extract(self, a, t, x_shape=(1, 1, 1, 1)):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def p_mean_variance(self, y_t, t, phase, x_a, clip_denoised: bool, y_cond=None):
        noise_level = self.extract(
            getattr(self.denoise_fn, "gammas_" + phase), t, x_shape=(1, 1)
        ).to(y_t.device)

        # we need grads for reconstruction guidance
        y_t.requires_grad = True

        denoise_fn_input = torch.cat([y_cond, y_t], dim=1)

        noise = self.denoise_fn(denoise_fn_input, noise_level)

        y_0_hat = predict_start_from_noise(
            self.denoise_fn,
            y_t,
            t=t,
            noise=noise,
            phase=phase,
        )

        #### TODO add reconstruction guidance

        if x_a is not None:

            omega_t = 2.0

            alpha_t = self.extract(
                getattr(self.denoise_fn, "alphas_" + phase), t, x_shape=(1, 1)
            ).to(y_t.device)

            mse = nn.MSELoss()

            distance = mse(x_a, y_0_hat)

            grad = torch.autograd.grad(outputs=distance, inputs=y_t)[0]

            y_0_hat = y_0_hat - ((omega_t * alpha_t) / 2) * grad

        if clip_denoised:
            y_0_hat.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = q_posterior(
            self.denoise_fn, y_0_hat=y_0_hat, y_t=y_t, t=t, phase=phase
        )
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = self.default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    def p_sample(self, y_t, t, phase, x_a, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t,
            t=t,
            clip_denoised=clip_denoised,
            y_cond=y_cond,
            phase=phase,
            x_a=x_a,
        )
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def forward(self, y_0, y_cond, mask, noise):
        b, *_ = y_0.shape
        t = torch.randint(
            1, self.denoise_fn.num_timesteps_train, (b,), device=y_0.device
        ).long()

        gammas = self.denoise_fn.gammas_train

        gamma_t1 = self.extract(gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = self.extract(gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand(
            (b, 1), device=y_0.device
        ) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = self.default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise
        )

        if mask is not None:
            temp_mask = torch.clamp(mask, min=0.0, max=1.0)
            noise_hat = self.denoise_fn(
                torch.cat(
                    [y_cond, y_noisy * temp_mask + (1.0 - temp_mask) * y_0], dim=1
                ),
                sample_gammas,
            )
        else:
            noise_hat = self.denoise_fn(
                torch.cat([y_cond, y_noisy], dim=1), sample_gammas
            )

        return noise, noise_hat
