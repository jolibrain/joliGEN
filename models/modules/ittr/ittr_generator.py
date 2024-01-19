import os
import torch
import torch.nn.functional as F
from torch import nn
from torch import nn, einsum
from einops import rearrange, reduce, repeat

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def l2norm(t):
    return F.normalize(t, dim=-1)


# helper classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


# classes


class HPB(nn.Module):
    """Hybrid Perception Block"""

    def __init__(
        self,
        dim,
        dim_head=32,
        heads=8,
        ff_mult=4,
        attn_height_top_k=16,
        attn_width_top_k=16,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()

        self.attn = DPSA(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            height_top_k=attn_height_top_k,
            width_top_k=attn_width_top_k,
            dropout=attn_dropout,
        )

        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.attn_parallel_combine_out = nn.Conv2d(dim * 2, dim, 1)

        ff_inner_dim = dim * ff_mult

        self.ff = nn.Sequential(
            nn.Conv2d(dim, ff_inner_dim, 1),
            nn.InstanceNorm2d(ff_inner_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            Residual(
                nn.Sequential(
                    nn.Conv2d(
                        ff_inner_dim, ff_inner_dim, 3, padding=1, groups=ff_inner_dim
                    ),
                    nn.InstanceNorm2d(ff_inner_dim),
                    nn.GELU(),
                    nn.Dropout(ff_dropout),
                )
            ),
            nn.Conv2d(ff_inner_dim, dim, 1),
            nn.InstanceNorm2d(ff_inner_dim),
        )

    def forward(self, x):
        attn_branch_out = self.attn(x)
        conv_branch_out = self.dwconv(x)

        concatted_branches = torch.cat((attn_branch_out, conv_branch_out), dim=1)
        attn_out = self.attn_parallel_combine_out(concatted_branches) + x

        return self.ff(attn_out)


class DPSA(nn.Module):
    """Dual-pruned Self-attention Block"""

    def __init__(
        self, dim, height_top_k=16, width_top_k=16, dim_head=32, heads=8, dropout=0.0
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.norm = ChanLayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.height_top_k = height_top_k
        self.width_top_k = width_top_k

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        # fold out heads

        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> (b h) c x y", h=self.heads),
            (q, k, v),
        )

        # they used l2 normalized queries and keys, cosine sim attention basically

        q, k = map(l2norm, (q, k))

        # calculate whether to select and rank along height and width

        need_height_select_and_rank = self.height_top_k < h
        need_width_select_and_rank = self.width_top_k < w

        # select and rank keys / values, probing with query (reduced along height and width) and keys reduced along row and column respectively

        if need_width_select_and_rank or need_height_select_and_rank:
            q_probe = reduce(q, "b h w d -> b d", "sum")

        # gather along height, then width

        if need_height_select_and_rank:
            k_height = reduce(k, "b h w d -> b h d", "sum")

            top_h_indices = (
                einsum("b d, b h d -> b h", q_probe, k_height)
                .topk(k=self.height_top_k, dim=-1)
                .indices
            )

            top_h_indices = repeat(
                top_h_indices, "b h -> b h w d", d=self.dim_head, w=k.shape[-2]
            )

            k, v = map(
                lambda t: t.gather(1, top_h_indices), (k, v)
            )  # first gather across height

        if need_width_select_and_rank:
            k_width = reduce(k, "b h w d -> b w d", "sum")

            top_w_indices = (
                einsum("b d, b w d -> b w", q_probe, k_width)
                .topk(k=self.width_top_k, dim=-1)
                .indices
            )

            top_w_indices = repeat(
                top_w_indices, "b w -> b h w d", d=self.dim_head, h=k.shape[1]
            )

            k, v = map(
                lambda t: t.gather(2, top_w_indices), (k, v)
            )  # then gather along width

        # select the appropriate keys and values

        q, k, v = map(lambda t: rearrange(t, "b ... d -> b (...) d"), (q, k, v))

        # cosine similarities

        sim = einsum("b i d, b j d -> b i j", q, k)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate out

        out = einsum("b i j, b j d -> b i d", attn, v)

        # merge heads and combine out

        out = rearrange(out, "(b h) (x y) d -> b (h d) x y", x=h, y=w, h=self.heads)
        return self.to_out(out)


class ITTRStem(nn.Module):
    """ITTR Stem"""

    def __init__(self, input_nc, ngf):
        super().__init__()

        self.ff = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.GELU(),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            self.ff += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.GELU(),
            ]
        self.ff = nn.Sequential(*self.ff)

    def forward(self, x):
        return self.ff(x)


class ITTRDecoder(nn.Module):
    """ITTR Decoder"""

    def __init__(self, output_nc, ngf):
        super().__init__()

        self.decoder = []
        n_upsampling = 2

        for i in range(n_upsampling):
            mult = 2 ** (n_upsampling - i)
            self.decoder += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.GELU(),
            ]
        self.decoder += [nn.ReflectionPad2d(3)]
        self.decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.decoder += [nn.Tanh()]
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        return self.decoder(x)


class ITTRGenerator(nn.Module):
    """ITTR generator that consists of a stem, a tranformer-like HDB module, and a decoder
    See https://arxiv.org/pdf/2203.16015.pdf
    """

    def __init__(self, input_nc, output_nc, img_size, n_blocks=9, ngf=64):
        assert n_blocks >= 0
        super(ITTRGenerator, self).__init__()

        # fixed options (for now)
        dim = ngf * 4  # ngf * 2 * 2 du to downsampling
        dim_head = int(img_size / 4)  # equal to feature map size out of stem
        heads = 8
        attn_height_top_k = 16
        attn_width_top_k = 16
        attn_dropout = 0.0
        ff_mult = 4
        ff_dropout = 0.0

        # networks
        self.stem = ITTRStem(input_nc, ngf)

        self.ittr = []
        for i in range(n_blocks):
            self.ittr.append(
                HPB(
                    dim,
                    dim_head,
                    heads,
                    ff_mult,
                    attn_height_top_k,
                    attn_width_top_k,
                    attn_dropout,
                    ff_dropout,
                )
            )
        self.ittr = nn.Sequential(*self.ittr)

        self.decoder = ITTRDecoder(output_nc, ngf)

    def compute_feats(self, x, extract_layer_ids=[]):
        feats = []
        out = self.stem(x)
        for i, hpb in enumerate(self.ittr):
            out = hpb(out)
            if i in extract_layer_ids:
                feats.append(out)
        return out, feats

    def get_feats(self, x, extract_layer_ids=[]):
        _, feats = self.compute_feats(x, extract_layer_ids)
        return feats

    def forward(self, x):
        out, _ = self.compute_feats(x)
        out = self.decoder(out)
        return out
