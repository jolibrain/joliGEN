import math
import sys
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from torch import nn
from einops import rearrange, repeat

from models.modules.diffusion_utils import (
    set_new_noise_schedule,
    predict_start_from_noise,
    q_posterior,
    gamma_embedding,
    extract,
    rearrange_5dto4d_fh,
    rearrange_4dto5d_fh,
    expand_for_video,
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
            self.cond_embed_gammas = self.denoise_fn.cond_embed_gammas

            self.cond_embed = nn.Sequential(
                nn.Linear(self.cond_embed_gammas, self.cond_embed_gammas),
                torch.nn.SiLU(),
                nn.Linear(self.cond_embed_gammas, self.cond_embed_gammas),
            )

            self.cond_embed_gammas_in = self.cond_embed_gammas

    # backward process
    def restoration(
        self,
        y_cond,
        y_t=None,
        y_0=None,
        mask=None,
        sample_num=8,
        cls=None,
        ref=None,
        guidance_scale=0.0,
        ddim_num_steps=10,
        ddim_eta=0.5,
    ):
        if self.sampling_method == "ddpm":
            return self.restoration_ddpm(
                y_cond,
                y_t=y_t,
                y_0=y_0,
                mask=mask,
                sample_num=sample_num,
                cls=cls,
                guidance_scale=guidance_scale,
                ref=ref,
            )
        elif self.sampling_method == "ddim":
            return self.restoration_ddim(
                y_cond,
                y_t=y_t,
                y_0=y_0,
                mask=mask,
                sample_num=sample_num,
                cls=cls,
                guidance_scale=guidance_scale,
                num_steps=ddim_num_steps,
                eta=ddim_eta,
                ref=ref,
            )

    ## DDPM
    def restoration_ddpm(
        self,
        y_cond,
        y_t,
        y_0,
        mask,
        sample_num,
        cls,
        guidance_scale,
        ref,
    ):
        phase = "test"

        b, *_ = y_cond.shape

        assert (
            self.denoise_fn.model.num_timesteps_test > sample_num
        ), "num_timesteps must greater than sample_num"
        sample_inter = self.denoise_fn.model.num_timesteps_test // sample_num

        # y_t must be of output channel size, since we do not have y_0 (gt), we get it from the model
        y_t_shape = list(y_cond.shape)
        y_t_shape[1] = (
            self.denoise_fn.model.out_channel
        )  # set to number of model output channels
        y_t = self.default(
            y_t,
            lambda: torch.randn(y_t_shape, device=y_cond.device, dtype=y_cond.dtype),
        )
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
                ref=ref,
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
        ref,
        y_cond=None,
        guidance_scale=0.0,
    ):
        sequence_length = 0
        if len(y_t.shape) == 5:
            sequence_length = y_t.shape[1]
            y_t, y_cond, mask = rearrange_5dto4d_fh(y_t, y_cond, mask)
        noise_level = self.extract(
            getattr(self.denoise_fn.model, "gammas_" + phase), t, x_shape=(1, 1)
        ).to(y_t.device)

        embed_noise_level = self.compute_gammas(noise_level)

        input = torch.cat([y_cond, y_t], dim=1)
        if sequence_length != 0:
            input, y_t, mask = rearrange_4dto5d_fh(sequence_length, input, y_t, mask)
        if guidance_scale > 0.0 and phase == "test":
            y_0_hat_uncond = predict_start_from_noise(
                self.denoise_fn.model,
                y_t,
                t=t,
                noise=self.denoise_fn(
                    input,
                    torch.zeros_like(embed_noise_level),
                    cls=None,
                    mask=None,
                    ref=ref,
                ),
                phase=phase,
            )
        y_0_hat = predict_start_from_noise(
            self.denoise_fn.model,
            y_t,
            t=t,
            noise=self.denoise_fn(
                input, embed_noise_level, cls=cls, mask=mask, ref=ref
            ),
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
        ref,
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
            ref=ref,
            guidance_scale=guidance_scale,
        )
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)

        out = model_mean + noise * (0.5 * model_log_variance).exp()

        return out

    ## DDIM
    def restoration_ddim(
        self,
        y_cond,
        y_t=None,
        y_0=None,
        mask=None,
        sample_num=8,
        cls=None,
        guidance_scale=0.0,
        num_steps=10,
        eta=0.5,
        ref=None,
    ):
        phase = "test"

        b, *_ = y_cond.shape

        assert (
            self.denoise_fn.model.num_timesteps_test > sample_num
        ), "num_timesteps must greater than sample_num"
        sample_inter = self.denoise_fn.model.num_timesteps_test // sample_num

        y_t = self.default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t

        # linear
        tseq = list(
            np.linspace(
                0, self.denoise_fn.model.num_timesteps_test - 1, num_steps
            ).astype(int)
        )

        tlist = torch.zeros([y_t.shape[0]], device=y_t.device).long()
        for i in tqdm(
            range(num_steps),
            desc="sampling loop time step",
            total=num_steps,
        ):
            tlist = tlist * 0 + tseq[-1 - i]

            if i != num_steps - 1:
                prevt = torch.ones_like(tlist, device=y_cond.device) * tseq[-2 - i]
            else:
                prevt = -torch.ones_like(tlist, device=y_cond.device)

            y_t = self.ddim_p_sample(
                y_t,
                tlist,
                prevt,
                y_cond=y_cond,
                phase=phase,
                cls=cls,
                mask=mask,
                ref=ref,
                guidance_scale=guidance_scale,
            )

            if mask is not None:
                temp_mask = torch.clamp(mask, min=0.0, max=1.0)
                y_t = y_0 * (1.0 - temp_mask) + temp_mask * y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def ddim_p_sample(
        self,
        y_t,
        t,
        prevt,
        phase,
        cls,
        mask,
        ref,
        clip_denoised=True,
        y_cond=None,
        guidance_scale=0.0,
    ):
        model_mean, model_log_variance = self.ddim_p_mean_variance(
            y_t=y_t,
            t=t,
            prevt=prevt,
            clip_denoised=clip_denoised,
            y_cond=y_cond,
            phase=phase,
            cls=cls,
            mask=mask,
            ref=ref,
            guidance_scale=guidance_scale,
        )

        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        out = model_mean
        return out

    def ddim_p_mean_variance(
        self,
        y_t,
        t,
        prevt,
        phase,
        clip_denoised: bool,
        cls,
        mask,
        ref,
        y_cond=None,
        guidance_scale=0.0,
        num_steps=10,
        eta=0.5,
    ):
        sequence_length = 0
        if len(y_t.shape) == 5:
            sequence_length = y_t.shape[1]
            y_t, y_cond, mask = rearrange_5dto4d_fh(y_t, y_cond, mask)
        noise_level = self.extract(
            getattr(self.denoise_fn.model, "gammas_" + phase), t, x_shape=(1, 1)
        ).to(y_t.device)

        embed_noise_level = self.compute_gammas(noise_level)

        input = torch.cat([y_cond, y_t], dim=1)

        if sequence_length != 0:
            input, y_t, mask = rearrange_4dto5d_fh(sequence_length, input, y_t, mask)
        if guidance_scale > 0.0 and phase == "test":
            y_0_hat_uncond = self.denoise_fn(
                input, embed_noise_level, cls=None, mask=None
            )

        y_0_hat = self.denoise_fn(input, embed_noise_level, cls=cls, mask=mask, ref=ref)
        if guidance_scale > 0.0 and phase == "test":
            y_0_hat = (1 + guidance_scale) * y_0_hat - guidance_scale * y_0_hat_uncond

        if clip_denoised:
            y_0_hat.clamp_(-1.0, 1.0)
        gamma_t = self.extract(
            getattr(self.denoise_fn.model, "gammas_" + phase), t, x_shape=(1, 1, 1, 1)
        ).to(y_t.device)
        gamma_prevt = self.extract(
            getattr(self.denoise_fn.model, "gammas_prev_" + phase),
            prevt + 1,
            x_shape=(1, 1, 1, 1),
        ).to(y_t.device)
        ## denoising formula for model_mean witih DDIM
        sigma = eta * torch.sqrt(
            (1 - gamma_prevt) / (1 - gamma_t) * (1 - gamma_t / gamma_prevt)
        )
        p_var = sigma**2
        posterior_log_variance = torch.log(p_var)

        coef_eps = 1 - gamma_prevt - p_var
        coef_eps[coef_eps < 0] = 0
        coef_eps = torch.sqrt(coef_eps)
        if (
            len(y_t.shape) == 5 and y_t.shape[0] > 1
        ):  # ddpm training and ddim inference for vid
            gamma_t = expand_for_video(gamma_t, y_t)
            gamma_prevt = expand_for_video(gamma_prevt, y_t)
            sigma = expand_for_video(sigma, y_t)
            coef_eps = expand_for_video(coef_eps, y_t)

        model_mean = (
            torch.sqrt(gamma_prevt)
            * (y_t - torch.sqrt(1.0 - gamma_t) * y_0_hat)
            / torch.sqrt(gamma_t)
            + coef_eps * y_0_hat
        )

        if clip_denoised:
            model_mean.clamp_(-1.0, 1.0)
        return model_mean, posterior_log_variance

    def forward(self, y_0, y_cond, mask, noise, cls, ref, dropout_prob=0.0):
        sequence_length = 0

        # vid only
        if len(y_0.shape) == 5:
            sequence_length = y_0.shape[1]
            y_0, y_cond, mask = rearrange_5dto4d_fh(y_0, y_cond, mask)

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

        if sequence_length != 0:
            input, mask, noise = rearrange_4dto5d_fh(
                sequence_length, input, mask, noise
            )

        noise_hat = self.denoise_fn(
            input, embed_sample_gammas, cls=cls, mask=mask, ref=ref
        )

        # min-SNR loss weight
        phase = "train"
        ksnr = 5.0
        snr1 = extract(
            getattr(self.denoise_fn.model, "sqrt_recip_gammas_" + phase), t, y_0.shape
        )
        snr2 = extract(
            getattr(self.denoise_fn.model, "sqrt_recipm1_gammas_" + phase), t, y_0.shape
        )

        snr = torch.pow(snr1 / snr2, 2).squeeze()
        if len(snr.shape) == 0:  # unsqueeze if batch size is 1
            snr = snr.unsqueeze(0)
        min_snr_loss_weight = (
            torch.stack([snr, ksnr * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
        )
        # reshape min_snr_loss_weight to match noise_hat
        min_snr_loss_weight = min_snr_loss_weight.view(-1, 1, 1, 1)

        return noise, noise_hat, min_snr_loss_weight

    def set_new_sampling_method(self, sampling_method):
        self.sampling_method = sampling_method

    def compute_gammas(self, gammas):
        emb = self.cond_embed(gamma_embedding(gammas, self.cond_embed_gammas_in))
        return emb
