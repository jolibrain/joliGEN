import math
import sys
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from torch import nn


from models.modules.diffusion_utils import (
    set_new_noise_schedule,
    predict_start_from_noise,
    q_posterior,
    gamma_embedding,
)


class DiffusionGenerator(nn.Module):
    def __init__(
        self,
        denoise_fn,
        sampling_method,
        image_size,
        G_ngf,
        loading_backward_compatibility,
    ):
        super().__init__()

        self.denoise_fn = denoise_fn
        self.sampling_method = sampling_method
        self.image_size = image_size

        cond_embed_dim = self.denoise_fn.cond_embed_dim

        # Init noise schedule
        set_new_noise_schedule(model=self.denoise_fn.model, phase="train")
        set_new_noise_schedule(model=self.denoise_fn.model, phase="test")

        # Backward compatibility

        if loading_backward_compatibility:
            if type(self.denoise_fn.model).__name__ == "ResnetGenerator_attn_diff":

                inner_channel = G_ngf
                self.cond_embed = nn.Sequential(
                    nn.Linear(inner_channel, cond_embed_dim),
                    torch.nn.SiLU(),
                    nn.Linear(cond_embed_dim, cond_embed_dim),
                )

            elif type(self.denoise_fn.model).__name__ == "UNet":

                inner_channel = G_ngf
                cond_embed_dim = inner_channel * 4

                self.cond_embed = nn.Sequential(
                    nn.Linear(inner_channel, cond_embed_dim),
                    torch.nn.SiLU(),
                    nn.Linear(cond_embed_dim, cond_embed_dim),
                )

            self.cond_embed_gammas_in = inner_channel
        else:
            self.cond_embed_dim = cond_embed_dim

            if "class" in self.denoise_fn.conditioning:
                self.cond_embed_gammas = self.cond_embed_dim // 2
            else:
                self.cond_embed_gammas = self.cond_embed_dim

            self.cond_embed = nn.Sequential(
                nn.Linear(self.cond_embed_gammas, self.cond_embed_gammas),
                torch.nn.SiLU(),
                nn.Linear(self.cond_embed_gammas, self.cond_embed_gammas),
            )

            self.cond_embed_gammas_in = self.cond_embed_gammas

    def restoration(
        self,
        y_cond,
        y_t=None,
        y_0=None,
        mask=None,
        sample_num=8,
        cls=None,
        guidance_scale=0.0,
    ):
        phase = "test"

        b, *_ = y_cond.shape

        assert (
            self.denoise_fn.model.num_timesteps_test > sample_num
        ), "num_timesteps must greater than sample_num"
        sample_inter = self.denoise_fn.model.num_timesteps_test // sample_num

        y_t = self.default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(
            reversed(range(0, self.denoise_fn.model.num_timesteps_test)),
            desc="sampling loop time step",
            total=self.denoise_fn.model.num_timesteps_test,
        ):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)

            y_t = self.p_sample(
                y_t,
                t,
                y_cond=y_cond,
                phase=phase,
                cls=cls,
                mask=mask,
                guidance_scale=guidance_scale,
            )

            if mask is not None:
                temp_mask = torch.clamp(mask, min=0.0, max=1.0)
                y_t = y_0 * (1.0 - temp_mask) + temp_mask * y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
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

    def p_mean_variance(
        self,
        y_t,
        t,
        phase,
        clip_denoised: bool,
        cls,
        mask,
        y_cond=None,
        guidance_scale=0.0,
    ):
        noise_level = self.extract(
            getattr(self.denoise_fn.model, "gammas_" + phase), t, x_shape=(1, 1)
        ).to(y_t.device)

        embed_noise_level = self.compute_gammas(noise_level)

        input = torch.cat([y_cond, y_t], dim=1)

        if guidance_scale > 0.0 and phase == "test":
            y_0_hat_uncond = predict_start_from_noise(
                self.denoise_fn.model,
                y_t,
                t=t,
                noise=self.denoise_fn(
                    input, torch.zeros_like(embed_noise_level), cls=None, mask=None
                ),
                phase=phase,
            )

        y_0_hat = predict_start_from_noise(
            self.denoise_fn.model,
            y_t,
            t=t,
            noise=self.denoise_fn(input, embed_noise_level, cls=cls, mask=mask),
            phase=phase,
        )

        if guidance_scale > 0.0 and phase == "test":
            y_0_hat = (1 + guidance_scale) * y_0_hat - guidance_scale * y_0_hat_uncond

        if clip_denoised:
            y_0_hat.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = q_posterior(
            self.denoise_fn.model, y_0_hat=y_0_hat, y_t=y_t, t=t, phase=phase
        )
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = self.default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    def p_sample(
        self,
        y_t,
        t,
        phase,
        cls,
        mask,
        clip_denoised=True,
        y_cond=None,
        guidance_scale=0.0,
    ):

        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t,
            t=t,
            clip_denoised=clip_denoised,
            y_cond=y_cond,
            phase=phase,
            cls=cls,
            mask=mask,
            guidance_scale=guidance_scale,
        )

        if self.sampling_method == "ddpm":
            noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        elif self.sampling_method == "ddim":
            noise = torch.zeros_like(y_t)
        else:
            raise ValueError(
                "%s sampling method is not implemented" % self.sampling_method
            )
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def forward(self, y_0, y_cond, mask, noise, cls, dropout_prob=0.0):

        b, *_ = y_0.shape
        t = torch.randint(
            1, self.denoise_fn.model.num_timesteps_train, (b,), device=y_0.device
        ).long()

        gammas = self.denoise_fn.model.gammas_train

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

        embed_sample_gammas = self.compute_gammas(sample_gammas)

        if mask is not None:
            temp_mask = torch.clamp(mask, min=0.0, max=1.0)
            y_noisy = y_noisy * temp_mask + (1.0 - temp_mask) * y_0

        input = torch.cat([y_cond, y_noisy], dim=1)

        noise_hat = self.denoise_fn(input, embed_sample_gammas, cls=cls, mask=mask)

        return noise, noise_hat

    def set_new_sampling_method(self, sampling_method):
        self.sampling_method = sampling_method

    def compute_gammas(self, gammas):
        emb = self.cond_embed(gamma_embedding(gammas, self.cond_embed_gammas_in))
        return emb
