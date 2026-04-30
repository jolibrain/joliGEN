import torch
from torch import nn
from torch.func import jvp, vmap

from .vit.vit import JiT


class _RMSNormNoAffine(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x_float = x.float()
        norm = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_float = x_float * norm
        return x_float.to(dtype)


class CAFMJiTDiscriminator(JiT):
    def __init__(self, flip_t=False, **kwargs):
        super().__init__(**kwargs)
        del self.final_layer
        self.flip_t = flip_t
        self.proj_out = nn.Sequential(
            _RMSNormNoAffine(kwargs["hidden_size"]),
            nn.Linear(kwargs["hidden_size"], 1),
        )

    def forward(self, x, y, t):
        if self.flip_t:
            t = 1.0 - t

        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(
                x,
                c,
                (
                    self.feat_rope
                    if i < self.in_context_start
                    else self.feat_rope_incontext
                ),
            )

        x = x[:, 0]
        x = self.proj_out(x)
        return x.reshape(-1)


class CAFMDiscriminatorJVP(nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.dis = discriminator

    def forward(self, x, y, t, dx, dt):
        def dis_fn(x_arg, t_arg):
            return self.dis(x_arg, y, t_arg)

        def dis_jvp(dx_arg, dt_arg):
            return jvp(dis_fn, (x, t), (dx_arg, dt_arg))

        if x.ndim == dx.ndim:
            return dis_jvp(dx, dt)
        return vmap(dis_jvp)(dx, dt)
