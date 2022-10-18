import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from .unet_generator_attn import UNet
from torch import nn


class DiffusionGenerator(nn.Module):
    def __init__(
        self,
        unet,
        image_size,
        in_channel,
        inner_channel,
        out_channel,
        res_blocks,
        attn_res,
        tanh,
        n_timestep_train,
        n_timestep_test,
        dropout=0,
        channel_mults=(1, 2, 4, 8),
        conv_resample=True,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    ):

        super().__init__()

        if unet == "unet_mha":
            self.denoise_fn = UNet(
                image_size=image_size,
                in_channel=in_channel,
                inner_channel=inner_channel,
                out_channel=out_channel,
                res_blocks=res_blocks,
                attn_res=attn_res,
                tanh=tanh,
                n_timestep_train=n_timestep_train,
                n_timestep_test=n_timestep_test,
                dropout=dropout,
                channel_mults=channel_mults,
                conv_resample=conv_resample,
                use_checkpoint=use_checkpoint,
                use_fp16=use_fp16,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                num_heads_upsample=num_heads_upsample,
                use_scale_shift_norm=use_scale_shift_norm,
                resblock_updown=resblock_updown,
                use_new_attention_order=use_new_attention_order,
            )

        # Init noise schedule
        self.denoise_fn.set_new_noise_schedule(phase="train")
        self.denoise_fn.set_new_noise_schedule(phase="test")

    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
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
            y_t = self.p_sample(y_t, t, y_cond=y_cond, phase=phase)
            if mask is not None:
                y_t = y_0 * (1.0 - mask) + mask * y_t
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

    def p_mean_variance(self, y_t, t, phase, clip_denoised: bool, y_cond=None):
        noise_level = self.extract(
            getattr(self.denoise_fn, "gammas_" + phase), t, x_shape=(1, 1)
        ).to(y_t.device)
        y_0_hat = self.denoise_fn.predict_start_from_noise(
            y_t,
            t=t,
            noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level),
            phase=phase,
        )

        if clip_denoised:
            y_0_hat.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.denoise_fn.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t, phase=phase
        )
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = self.default(noise, lambda: torch.randn_like(y_0))
        return sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise

    def p_sample(self, y_t, t, phase, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond, phase=phase
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
            noise_hat = self.denoise_fn(
                torch.cat([y_cond, y_noisy * mask + (1.0 - mask) * y_0], dim=1),
                sample_gammas,
            )
        else:
            noise_hat = self.denoise_fn(
                torch.cat([y_cond, y_noisy], dim=1), sample_gammas
            )

        return noise, noise_hat
