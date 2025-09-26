import numpy as np
import torch
import math
from functools import partial
from einops import rearrange


def gamma_embedding_1D(gammas, dim, max_period):
    """
    Create sinusoidal timestep embeddings.
    :param gammas: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


def gamma_embedding(gammas, dim, max_period=10000):

    return_list = []
    reduced_dim = dim // gammas.shape[1]

    for i in range(gammas.shape[1]):
        return_list.append(
            gamma_embedding_1D(gammas[:, i], reduced_dim, max_period=max_period)
        )
    embedding = torch.cat(return_list, dim=1)

    return embedding


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3
):
    if schedule == "quad":
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64
            )
            ** 2
        )
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


def set_new_noise_schedule(model, phase):
    param = next(model.parameters(), None)
    if param is not None:
        device = param.device
    else:
        buf = next(model.buffers(), None)
        device = buf.device if buf is not None else torch.device("cpu")

    to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
    betas = make_beta_schedule(**model.beta_schedule[phase])
    betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
    alphas = 1.0 - betas

    (timesteps,) = betas.shape
    setattr(model, "num_timesteps_" + phase, int(timesteps))
    gammas = np.cumprod(alphas, axis=0)
    gammas_prev = np.append(1.0, gammas[:-1])

    # calculations for diffusion q(x_t | x_{t-1}) and others
    model.register_buffer("gammas_" + phase, to_torch(gammas))
    model.register_buffer("gammas_prev_" + phase, to_torch(gammas_prev))
    model.register_buffer("sqrt_recip_gammas_" + phase, to_torch(np.sqrt(1.0 / gammas)))
    model.register_buffer(
        "sqrt_recipm1_gammas_" + phase, to_torch(np.sqrt(1.0 / gammas - 1))
    )

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - gammas_prev) / (1.0 - gammas)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    model.register_buffer(
        "posterior_log_variance_clipped_" + phase,
        to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
    )
    model.register_buffer(
        "posterior_mean_coef1_" + phase,
        to_torch(betas * np.sqrt(gammas_prev) / (1.0 - gammas)),
    )
    model.register_buffer(
        "posterior_mean_coef2_" + phase,
        to_torch((1.0 - gammas_prev) * np.sqrt(alphas) / (1.0 - gammas)),
    )


def predict_start_from_noise(model, y_t, t, noise, phase):
    return (
        extract(getattr(model, "sqrt_recip_gammas_" + phase), t, y_t.shape) * y_t
        - extract(getattr(model, "sqrt_recipm1_gammas_" + phase), t, y_t.shape) * noise
    )


def q_posterior(model, y_0_hat, y_t, t, phase):
    posterior_mean = (
        extract(getattr(model, "posterior_mean_coef1_" + phase), t, y_t.shape) * y_0_hat
        + extract(getattr(model, "posterior_mean_coef2_" + phase), t, y_t.shape) * y_t
    )
    posterior_log_variance_clipped = extract(
        getattr(model, "posterior_log_variance_clipped_" + phase), t, y_t.shape
    )
    return posterior_mean, posterior_log_variance_clipped


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def rearrange_5dto4d_fh(*tensors):
    """Rearrange a tensor according to a given pattern using einops.rearrange."""
    return [rearrange(tensor, "b f c h w -> b c (f h) w") for tensor in tensors]


def rearrange_4dto5d_fh(frame, *tensors):
    """Rearrange a tensor from 4D to 5D according to a given pattern using einops.rearrange."""
    return [
        rearrange(tensor, "b c (f h) w -> b f c h w", f=frame) for tensor in tensors
    ]


def rearrange_5dto4d_bf(*tensors):
    """Rearrange a tensor according to a given pattern using einops.rearrange."""
    return [rearrange(tensor, "b f c h w -> (b f) c h w") for tensor in tensors]


def rearrange_4dto5d_bf(frame, *tensors):
    """Rearrange a tensor from 4D to 5D according to a given pattern using einops.rearrange."""
    return [
        rearrange(tensor, "(b f) c h w -> b f c h w", f=frame) for tensor in tensors
    ]


def expand_for_video(coeff, y_t):
    B, T, C, H, W = y_t.shape
    coeff = coeff.view(B, 1, 1, 1, 1)  # start as per-batch scalar
    coeff = coeff.expand(B, T, 1, 1, 1)  # repeat along time
    return coeff
