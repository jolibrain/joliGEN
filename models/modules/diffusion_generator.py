import math
import sys
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from torch import nn
from einops import rearrange

from models.modules.unet_generator_attn.unet_generator_attn import UNet
from models.modules.diffusion_utils import (
    set_new_noise_schedule,
    predict_start_from_noise,
    q_posterior,
)

from models.modules.resnet_architecture.resnet_generator_diff import (
    ResnetGenerator_attn_diff,
)

from models.modules.diffusion_utils import gamma_embedding


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size):
        super().__init__()

        self.embedding_table = nn.Embedding(
            num_classes,
            hidden_size,
            max_norm=1.0,
            scale_grad_by_freq=True,
        )
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


class DiffusionGenerator(nn.Module):
    def __init__(
        self,
        denoise_fn,
        sampling_method,
        conditioning,
        num_classes,
        cond_embed_dim,
        image_size,
    ):

        super().__init__()

        self.denoise_fn = denoise_fn
        self.conditioning = conditioning

        self.sampling_method = sampling_method

        self.image_size = image_size

        # Init noise schedule
        set_new_noise_schedule(model=self.denoise_fn, phase="train")
        set_new_noise_schedule(model=self.denoise_fn, phase="test")

        # Label embedding

        if "class" in self.conditioning:
            self.cond_embed_dim = cond_embed_dim // 2

            cond_embed_class = self.cond_embed_dim

            self.l_embedder_class = LabelEmbedder(
                num_classes, self.cond_embed_dim  # * image_size * image_size
            )
            nn.init.normal_(self.l_embedder_class.embedding_table.weight, std=0.02)

        else:
            self.cond_embed_dim = cond_embed_dim

        if "mask" in self.conditioning:

            cond_embed_mask = cond_embed_dim

            self.l_embedder_mask = LabelEmbedder(
                num_classes, cond_embed_mask  # * image_size * image_size
            )
            nn.init.normal_(self.l_embedder_mask.embedding_table.weight, std=0.02)

        self.cond_embed = nn.Sequential(
            nn.Linear(self.cond_embed_dim, self.cond_embed_dim),
            torch.nn.SiLU(),
            nn.Linear(self.cond_embed_dim, self.cond_embed_dim),
        )

    def restoration(
        self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, cls=None
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

            if "mask" in self.conditioning:
                mask_embed = mask.to(torch.int32).squeeze(1)
                mask_embed = rearrange(mask_embed, "b h w -> b (h w)")
                mask_embed = self.l_embedder_mask(mask_embed)
                mask_embed = rearrange(
                    mask_embed, "b (h w) c -> b c h w", h=self.image_size
                )
            else:
                mask_embed = None

            if "class" in self.conditioning:
                cls_embed = self.l_embedder_class(cls)
            else:
                cls_embed = None

            y_t = self.p_sample(
                y_t,
                t,
                y_cond=y_cond,
                phase=phase,
                mask_embed=mask_embed,
                cls_embed=cls_embed,
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
        self, y_t, t, phase, clip_denoised: bool, mask_embed, cls_embed, y_cond=None
    ):
        noise_level = self.extract(
            getattr(self.denoise_fn, "gammas_" + phase), t, x_shape=(1, 1)
        ).to(y_t.device)

        noise_level = self.compute_gammas(noise_level)

        if "class" in self.conditioning:
            noise_level = torch.cat((noise_level, cls_embed), dim=1)

        input = torch.cat([y_cond, y_t], dim=1)

        if "mask" in self.conditioning:
            input = torch.cat([input, mask_embed], dim=1)

        y_0_hat = predict_start_from_noise(
            self.denoise_fn,
            y_t,
            t=t,
            noise=self.denoise_fn(input, noise_level),
            phase=phase,
        )

        if clip_denoised:
            y_0_hat.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = q_posterior(
            self.denoise_fn, y_0_hat=y_0_hat, y_t=y_t, t=t, phase=phase
        )
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = self.default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    def p_sample(
        self, y_t, t, phase, cls_embed, mask_embed, clip_denoised=True, y_cond=None
    ):

        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t,
            t=t,
            clip_denoised=clip_denoised,
            y_cond=y_cond,
            phase=phase,
            cls_embed=cls_embed,
            mask_embed=mask_embed,
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

    def forward(self, y_0, y_cond, mask, noise, cls):
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

        sample_gammas = self.compute_gammas(sample_gammas)

        if mask is not None:
            temp_mask = torch.clamp(mask, min=0.0, max=1.0)
            y_noisy = y_noisy * temp_mask + (1.0 - temp_mask) * y_0

        input = torch.cat([y_cond, y_noisy], dim=1)

        if "class" in self.conditioning:
            cls_embed = self.l_embedder_class(cls)
            sample_gammas = torch.cat((sample_gammas, cls_embed), dim=1)

        if "mask" in self.conditioning:
            mask_embed = mask.to(torch.int32).squeeze(1)
            mask_embed = rearrange(mask_embed, "b h w -> b (h w)")
            mask_embed = self.l_embedder_mask(mask_embed)
            mask_embed = rearrange(
                mask_embed, "b (h w) c -> b c h w", h=self.image_size
            )

            input = torch.cat([input, mask_embed], dim=1)

        noise_hat = self.denoise_fn(input, sample_gammas)

        return noise, noise_hat

    def set_new_sampling_method(self, sampling_method):
        self.sampling_method = sampling_method

    def compute_gammas(self, gammas):

        emb = self.cond_embed(gamma_embedding(gammas, self.cond_embed_dim))

        return emb
