from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange
from einops.layers.torch import Rearrange

from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

from .unet_attn_utils import (
    checkpoint,
    zero_module,
    normalization,
    normalization1d,
    count_flops_attn,
)

from models.modules.diffusion_utils import gamma_embedding


class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """


class EmbedSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """

    def __init__(
        self, channels, use_conv, out_channel=None, efficient=False, freq_space=False
    ):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        self.freq_space = freq_space

        if freq_space:
            from ..freq_utils import InverseHaarTransform, HaarTransform

            self.iwt = InverseHaarTransform(3)
            self.dwt = HaarTransform(3)
            self.channels = int(self.channels / 4)
            self.out_channel = int(self.out_channel / 4)

        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channel, 3, padding=1)
        self.efficient = efficient

    def forward(self, x):
        if self.freq_space:
            x = self.iwt(x)

        assert x.shape[1] == self.channels
        if not self.efficient:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        if self.efficient:  # if efficient, we do the interpolation after the conv
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        if self.freq_space:
            x = self.dwt(x)

        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channel=None, freq_space=False):
        super().__init__()
        self.channels = channels
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        self.freq_space = freq_space

        if self.freq_space:
            from ..freq_utils import InverseHaarTransform, HaarTransform

            self.iwt = InverseHaarTransform(3)
            self.dwt = HaarTransform(3)
            self.channels = int(self.channels / 4)
            self.out_channel = int(self.out_channel / 4)

        stride = 2
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channel, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channel
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        if self.freq_space:
            x = self.iwt(x)

        assert x.shape[1] == self.channels
        opx = self.op(x)

        if self.freq_space:
            opx = self.dwt(opx)

        return opx


class ResBlock(EmbedBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of embedding channels.
    :param dropout: the rate of dropout.
    :param out_channel: if specified, the number of out channels.
    :param use_conv: if True and out_channel is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        norm,
        out_channel=None,
        use_conv=False,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
        efficient=False,
        freq_space=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.up = up
        self.efficient = efficient
        self.freq_space = freq_space
        self.updown = up or down

        self.in_layers = nn.Sequential(
            normalization(self.channels, norm),
            torch.nn.SiLU(),
            nn.Conv2d(self.channels, self.out_channel, 3, padding=1),
        )

        if up:
            self.h_upd = Upsample(channels, False, freq_space=self.freq_space)
            self.x_upd = Upsample(channels, False, freq_space=self.freq_space)
        elif down:
            self.h_upd = Downsample(channels, False, freq_space=self.freq_space)
            self.x_upd = Downsample(channels, False, freq_space=self.freq_space)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            torch.nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channel if use_scale_shift_norm else self.out_channel,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channel, norm),
            torch.nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)),
        )

        if self.out_channel == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]

            h = in_rest(x)

            if self.efficient and self.up:
                h = in_conv(h)
                h = self.h_upd(h)
                x = self.x_upd(x)
            else:
                h = self.h_upd(h)
                x = self.x_upd(x)
                h = in_conv(h)

        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out.unsqueeze(-1)
            # emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        skipw = 1.0
        if self.efficient:
            skipw = 1.0 / math.sqrt(2)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        use_transformer=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.use_transformer = use_transformer
        self.norm = normalization1d(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        if self.use_transformer:
            x = x.reshape(b, -1, c)
        else:
            x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNet(nn.Module):
    """
    The full UNet model with attention and embedding.
    :param in_channel: channels in the input Tensor, for image colorization : Y_channels + X_channels .
    :param inner_channel: base channel count for the model.
    :param out_channel: channels in the output Tensor.
    :param res_blocks: number of residual blocks per downsample.
    :param attn_res: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mults: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channel,
        inner_channel,
        out_channel,
        res_blocks,
        attn_res,
        tanh,
        n_timestep_train,
        n_timestep_test,
        norm,
        group_norm_size,
        cond_embed_dim,
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
        efficient=False,
        freq_space=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.out_channel = out_channel
        self.res_blocks = res_blocks
        self.attn_res = attn_res
        self.dropout = dropout
        self.channel_mults = channel_mults
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.freq_space = freq_space

        if self.freq_space:
            from ..freq_utils import InverseHaarTransform, HaarTransform

            self.iwt = InverseHaarTransform(3)
            self.dwt = HaarTransform(3)
            in_channel *= 4
            out_channel *= 4

        if norm == "groupnorm":
            norm = norm + str(group_norm_size)

        self.cond_embed_dim = cond_embed_dim

        ch = input_ch = int(channel_mults[0] * self.inner_channel)
        self.input_blocks = nn.ModuleList(
            [EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        self.cond_embed_dim,
                        dropout,
                        out_channel=int(mult * self.inner_channel),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm=norm,
                        efficient=efficient,
                        freq_space=self.freq_space,
                    )
                ]
                ch = int(mult * self.inner_channel)
                if ds in attn_res:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock(
                            ch,
                            self.cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            norm=norm,
                            efficient=efficient,
                            freq_space=self.freq_space,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            out_channel=out_ch,
                            freq_space=self.freq_space,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = EmbedSequential(
            ResBlock(
                ch,
                self.cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                norm=norm,
                efficient=efficient,
                freq_space=self.freq_space,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                self.cond_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                norm=norm,
                efficient=efficient,
                freq_space=self.freq_space,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.cond_embed_dim,
                        dropout,
                        out_channel=int(self.inner_channel * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm=norm,
                        efficient=efficient,
                        freq_space=self.freq_space,
                    )
                ]
                ch = int(self.inner_channel * mult)
                if ds in attn_res:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.cond_embed_dim,
                            dropout,
                            out_channel=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            norm=norm,
                            efficient=efficient,
                            freq_space=self.freq_space,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            out_channel=out_ch,
                            freq_space=self.freq_space,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        if tanh:
            self.out = nn.Sequential(
                normalization(ch, norm),
                zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
                nn.Tanh(),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch, norm),
                torch.nn.SiLU(),
                zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
            )

        self.beta_schedule = {
            "train": {
                "schedule": "linear",
                "n_timestep": n_timestep_train,
                "linear_start": 1e-6,
                "linear_end": 0.01,
            },
            "test": {
                "schedule": "linear",
                "n_timestep": n_timestep_test,
                "linear_start": 1e-4,
                "linear_end": 0.09,
            },
        }

    def compute_feats(self, input, embed_gammas):
        if embed_gammas is None:
            # Only for GAN
            b = (input.shape[0], self.cond_embed_dim)
            embed_gammas = torch.ones(b).to(input.device)

        emb = embed_gammas

        hs = []

        h = input.type(torch.float32)

        if self.freq_space:
            h = self.dwt(h)

        for module in self.input_blocks:

            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)

        outs, feats = h, hs
        return outs, feats, emb

    def forward(self, input, embed_gammas=None):
        h, hs, emb = self.compute_feats(input, embed_gammas=embed_gammas)

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(input.dtype)
        outh = self.out(h)

        if self.freq_space:
            outh = self.iwt(outh)

        return outh

    def get_feats(self, input, extract_layer_ids):
        _, hs, _ = self.compute_feats(input, embed_gammas=None)
        feats = []

        for i, feat in enumerate(hs):
            if i in extract_layer_ids:
                feats.append(feat)

        return feats

    def extract(self, a, t, x_shape=(1, 1, 1, 1)):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# Transformer blocks
class LayerNorm(nn.Module):
    def __init__(self, dim, scale=True, normalize_dim=2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1

        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        normalize_dim = self.normalize_dim
        scale = (
            append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        )

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=normalize_dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=normalize_dim, keepdim=True)
        return (x - mean) * var.clamp(min=eps).rsqrt() * scale


class FeedForward(EmbedBlock):
    def __init__(self, dim, emb_dim, norm, mult=4, dropout=0.0):
        super().__init__()
        # self.norm = normalization(dim, norm)
        self.norm = LayerNorm(dim, scale=False)
        dim_hidden = dim * mult

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_dim, dim_hidden * 2), Rearrange("b d -> b 1 d")
        )

        to_scale_shift_linear = self.to_scale_shift[-2]
        nn.init.zeros_(to_scale_shift_linear.weight)
        nn.init.zeros_(to_scale_shift_linear.bias)

        self.proj_in = nn.Sequential(nn.Linear(dim, dim_hidden, bias=False), nn.SiLU())

        self.proj_out = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(dim_hidden, dim, bias=False)
        )

    def forward(self, x, emb):
        x = self.norm(x)
        x = self.proj_in(x)

        scale, shift = self.to_scale_shift(emb).chunk(2, dim=-1)
        x = x * (scale + 1) + shift

        return self.proj_out(x)


class UViT(nn.Module):
    """
    Implemented from 230111093
    """

    def __init__(
        self,
        image_size,
        in_channel,
        inner_channel,
        out_channel,
        res_blocks,
        attn_res,
        tanh,
        n_timestep_train,
        n_timestep_test,
        norm,
        group_norm_size,
        cond_embed_dim,
        dropout=0.2,
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
        num_transformer_blocks=6,
        efficient=False,
        freq_space=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.out_channel = out_channel
        self.res_blocks = res_blocks
        self.attn_res = attn_res
        self.dropout = dropout
        self.channel_mults = channel_mults
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.freq_space = freq_space

        if self.freq_space:
            from ..freq_utils import InverseHaarTransform, HaarTransform

            self.iwt = InverseHaarTransform(3)
            self.dwt = HaarTransform(3)
            in_channel *= 4
            out_channel *= 4

        if norm == "groupnorm":
            norm = norm + str(group_norm_size)

        self.cond_embed_dim = cond_embed_dim
        self.inner_channel = inner_channel

        ch = input_ch = int(channel_mults[0] * self.inner_channel)
        self.input_blocks = nn.ModuleList(
            [EmbedSequential(nn.Conv2d(in_channel, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        cond_embed_dim,
                        dropout,
                        out_channel=int(mult * self.inner_channel),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm=norm,
                        freq_space=self.freq_space,
                    )
                ]
                ch = int(mult * self.inner_channel)
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mults) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        Downsample(
                            ch,
                            conv_resample,
                            out_channel=out_ch,
                            freq_space=self.freq_space,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.learned_sinusoidal_dim = ch
        self.sinu_pos_emb = PositionalEncoding1D(self.learned_sinusoidal_dim)

        self.middle_blocks = nn.ModuleList()
        for _ in range(num_transformer_blocks):
            layers = [
                FeedForward(ch, cond_embed_dim, norm, dropout=dropout),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    use_new_attention_order=use_new_attention_order,
                    use_transformer=True,
                ),
            ]
            self.middle_blocks.append(EmbedSequential(*layers))

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        cond_embed_dim,
                        dropout,
                        out_channel=int(self.inner_channel * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        norm=norm,
                        freq_space=self.freq_space,
                    )
                ]
                ch = int(self.inner_channel * mult)
                if level and i == res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        Upsample(
                            ch,
                            conv_resample,
                            out_channel=out_ch,
                            efficient=efficient,
                            freq_space=self.freq_space,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        if tanh:
            self.out = nn.Sequential(
                normalization(ch, norm),
                zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
                nn.Tanh(),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch, norm),
                torch.nn.SiLU(),
                zero_module(nn.Conv2d(input_ch, out_channel, 3, padding=1)),
            )

        self.beta_schedule = {
            "train": {
                "schedule": "linear",
                "n_timestep": n_timestep_train,
                "linear_start": 1e-6,
                "linear_end": 0.01,
            },
            "test": {
                "schedule": "linear",
                "n_timestep": n_timestep_test,
                "linear_start": 1e-4,
                "linear_end": 0.09,
            },
        }

    def compute_feats(self, input, embed_gammas):
        if embed_gammas is None:
            # Only for GAN
            b = (input.shape[0], self.cond_embed_dim)
            embed_gammas = torch.ones(b).to(input.device)

        emb = embed_gammas

        hs = []

        h = input.type(torch.float32)

        if self.freq_space:
            h = self.dwt(h)

        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        B, C, H, W = h.shape
        h = h.reshape(B, H * W, C).to(input.device)
        hpos = self.sinu_pos_emb(h).to(input.device)
        h += hpos  # adding positional encoding
        for module in self.middle_blocks:
            h = module(h, emb)
        h = h.reshape(B, C, H, W)

        outs, feats = h, hs
        return outs, feats, emb

    def forward(self, input, embed_gammas=None):
        h, hs, emb = self.compute_feats(input, embed_gammas=embed_gammas)

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(input.dtype)
        outh = self.out(h)

        if self.freq_space:
            outh = self.iwt(outh)

        return outh

    def get_feats(self, input, extract_layer_ids):
        _, hs, _ = self.compute_feats(input, embed_gammas=None)
        feats = []

        for i, feat in enumerate(hs):
            if i in extract_layer_ids:
                feats.append(feat)

        return feats

    def extract(self, a, t, x_shape=(1, 1, 1, 1)):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))


if __name__ == "__main__":
    b, c, h, w = 3, 6, 64, 64
    timsteps = 100
    model = UNet(
        image_size=h,
        in_channel=c,
        inner_channel=64,
        out_channel=3,
        res_blocks=[2, 2, 2, 2],
        attn_res=[8],
    )
    x = torch.randn((b, c, h, w))
    emb = torch.ones((b,))
    out = model(x, emb)
