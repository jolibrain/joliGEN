import math
from typing import Literal
from urllib.parse import urlparse

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn
import torch.nn.functional as F

from models.modules.unet_generator_attn.unet_attn_utils import zero_module
from .vit import LabelEmbedder, TimestepEmbedder, modulate
from .vit_vid_per_layer_motion import MotionModule as SharedMotionModule


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))
        self.init_values = init_values
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x):
        return x * self.gamma


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.ones_like(self.bias))
            o = self.out_features
            self.bias_mask[o // 3 : 2 * o // 3] = 0

    def forward(self, x):
        bias = (
            self.bias * self.bias_mask.to(self.bias.dtype)
            if self.bias is not None
            else None
        )
        return F.linear(x, self.weight, bias)


def rope_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x, sin, cos):
    return (x * cos) + (rope_rotate_half(x) * sin)


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        *,
        num_heads,
        base=100.0,
        min_period=None,
        max_period=None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
        dtype=torch.float32,
    ):
        super().__init__()
        if embed_dim % (4 * num_heads) != 0:
            raise ValueError("EUPE RoPE requires embed_dim divisible by 4 * num_heads")
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Provide either base or min_period/max_period")
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = embed_dim // num_heads
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.register_buffer(
            "periods", torch.empty(self.D_head // 4, dtype=dtype), persistent=True
        )
        self._init_weights()

    def _init_weights(self):
        device = self.periods.device
        dtype = self.periods.dtype
        if self.base is not None:
            periods = self.base ** (
                2
                * torch.arange(self.D_head // 4, device=device, dtype=dtype)
                / (self.D_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.D_head // 4, device=device, dtype=dtype
            )
            periods = (base**exponents) / base * self.max_period
        self.periods.data.copy_(periods)

    def forward(self, *, H, W):
        device = self.periods.device
        dtype = self.periods.dtype
        dd = {"device": device, "dtype": dtype}
        if self.normalize_coords == "max":
            max_hw = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_hw
            coords_w = torch.arange(0.5, W, **dd) / max_hw
        elif self.normalize_coords == "min":
            min_hw = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_hw
            coords_w = torch.arange(0.5, W, **dd) / min_hw
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0
        if self.training and self.shift_coords is not None:
            coords += torch.empty(2, **dd).uniform_(
                -self.shift_coords, self.shift_coords
            )
        if self.training and self.jitter_coords is not None:
            jitter = (
                torch.empty(2, **dd)
                .uniform_(-np.log(self.jitter_coords), np.log(self.jitter_coords))
                .exp()
            )
            coords *= jitter[None, :]
        if self.training and self.rescale_coords is not None:
            rescale = (
                torch.empty(1, **dd)
                .uniform_(-np.log(self.rescale_coords), np.log(self.rescale_coords))
                .exp()
            )
            coords *= rescale
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        k = 1 / (self.proj.in_channels * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        mask_k_bias=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def apply_rope(q, k, rope):
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        prefix = q.shape[-2] - sin.shape[-2]
        q = torch.cat(
            (q[:, :, :prefix], rope_apply(q[:, :, prefix:], sin, cos)), dim=-2
        )
        k = torch.cat(
            (k[:, :, :prefix], rope_apply(k[:, :, prefix:], sin, cos)), dim=-2
        )
        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        layerscale_init=1e-5,
        norm_layer=None,
        mask_k_bias=True,
    ):
        super().__init__()
        norm_layer = norm_layer or (lambda d: nn.LayerNorm(d, eps=1e-5))
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
        )
        self.ls1 = (
            LayerScale(dim, layerscale_init) if layerscale_init else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * ffn_ratio),
            act_layer=nn.GELU,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, layerscale_init) if layerscale_init else nn.Identity()
        )

    def forward(self, x, rope=None):
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DiffusionFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-5)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


EUPE_VARIANT_CONFIGS = {
    "EUPE-T/16": dict(embed_dim=192, depth=12, num_heads=3, patch_size=16),
    "EUPE-S/16": dict(embed_dim=384, depth=12, num_heads=6, patch_size=16),
    "EUPE-B/16": dict(embed_dim=768, depth=12, num_heads=12, patch_size=16),
    "eupe_vitt16": dict(embed_dim=192, depth=12, num_heads=3, patch_size=16),
    "eupe_vits16": dict(embed_dim=384, depth=12, num_heads=6, patch_size=16),
    "eupe_vitb16": dict(embed_dim=768, depth=12, num_heads=12, patch_size=16),
}


def _is_url(path):
    parsed = urlparse(str(path))
    return parsed.scheme in ("http", "https", "file")


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "teacher", "student", "module"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    prefixes = ("teacher.", "student.", "module.")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _adapt_patch_weight(source, target):
    if source.shape == target.shape:
        return source
    if source.ndim != 4 or target.ndim != 4:
        return None
    if source.shape[0] != target.shape[0] or source.shape[2:] != target.shape[2:]:
        return None
    if source.shape[1] != 3 or target.shape[1] < 3:
        return None

    adapted = target.detach().clone()
    adapted.zero_()
    rgb_groups = max(1, target.shape[1] // 3)
    for start in range(0, target.shape[1] - 2, 3):
        adapted[:, start : start + 3] = source / rgb_groups
    return adapted


class EUPEDiffusionViT(nn.Module):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        out_channels=None,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        num_classes=1000,
        n_storage_tokens=4,
        pretrained_weights="",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.out_channel = self.out_channels
        self.patch_size = patch_size
        self.hidden_size = embed_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.input_size = input_size
        self.num_classes = num_classes
        self.n_storage_tokens = n_storage_tokens

        self.t_embedder = TimestepEmbedder(embed_dim)
        self.y_embedder = LabelEmbedder(num_classes, embed_dim)
        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="separate",
            rescale_coords=2,
            dtype=torch.float32,
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    embed_dim,
                    num_heads,
                    ffn_ratio=ffn_ratio,
                    layerscale_init=1e-5,
                    norm_layer=lambda d: nn.LayerNorm(d, eps=1e-5),
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.head = nn.Identity()
        self.diffusion_prefix = nn.Linear(embed_dim, (1 + n_storage_tokens) * embed_dim)
        self.final_layer = DiffusionFinalLayer(embed_dim, patch_size, self.out_channels)
        self.initialize_weights()
        if pretrained_weights:
            self.load_eupe_pretrained(pretrained_weights)

    def initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        nn.init.zeros_(self.diffusion_prefix.weight)
        nn.init.zeros_(self.diffusion_prefix.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def load_eupe_pretrained(self, weights):
        if _is_url(weights):
            state_dict = torch.hub.load_state_dict_from_url(weights, map_location="cpu")
        else:
            state_dict = torch.load(weights, map_location="cpu")
        state_dict = _unwrap_state_dict(state_dict)
        model_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            if key == "patch_embed.proj.weight" and key in model_state:
                adapted = _adapt_patch_weight(value, model_state[key])
                if adapted is not None:
                    filtered[key] = adapted
                else:
                    skipped.append(key)
                continue
            if key in model_state and value.shape == model_state[key].shape:
                filtered[key] = value
            elif key.startswith(
                (
                    "patch_embed.",
                    "cls_token",
                    "storage_tokens",
                    "blocks.",
                    "norm.",
                    "mask_token",
                    "rope_embed.",
                )
            ):
                skipped.append(key)
        loaded_block_keys = [key for key in filtered if key.startswith("blocks.")]
        if not loaded_block_keys:
            raise RuntimeError(
                f"No EUPE transformer block weights could be loaded from {weights}"
            )
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(
            f"Loaded EUPE pretrained weights: {len(filtered)}/{len(model_state)} keys; "
            f"{len(missing)} missing, {len(unexpected)} unexpected, {len(skipped)} skipped."
        )
        if skipped:
            print("Skipped EUPE keys with incompatible shapes:", sorted(skipped)[:20])

    def _tokens(self, x, c):
        x, (H, W) = self.patch_embed(x)
        B = x.shape[0]
        prefix = torch.cat(
            [self.cls_token.expand(B, -1, -1), self.storage_tokens.expand(B, -1, -1)],
            dim=1,
        )
        prefix = prefix + self.diffusion_prefix(c).view(
            B, 1 + self.n_storage_tokens, self.hidden_size
        )
        return torch.cat([prefix, x], dim=1), (H, W)

    def _unpatchify(self, x, B=None, F_frames=None):
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise RuntimeError(f"Patch token count {x.shape[1]} is not square")
        x = x.reshape(x.shape[0], h, w, self.patch_size, self.patch_size, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], c, h * self.patch_size, w * self.patch_size)
        if B is not None and F_frames is not None:
            imgs = rearrange(imgs, "(b f) c h w -> b f c h w", b=B, f=F_frames)
        return imgs

    def forward_tokens(self, x, c):
        x, (H, W) = self._tokens(x, c)
        rope = self.rope_embed(H=H, W=W)
        for block in self.blocks:
            x = block(x, rope)
        x = self.norm(x)
        return x[:, 1 + self.n_storage_tokens :]

    def forward(self, x, t, y):
        c = self.t_embedder(t.reshape(-1)) + self.y_embedder(y.reshape(-1))
        x = self.forward_tokens(x, c)
        x = self.final_layer(x, c)
        return self._unpatchify(x)


class EUPEDiffusionViTVideo(EUPEDiffusionViT):
    def __init__(
        self,
        *args,
        max_frames=8,
        motion_num_heads=8,
        motion_num_layers=2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_frames = max_frames
        self.motion_module = SharedMotionModule(
            in_channels=self.hidden_size,
            num_attention_heads=motion_num_heads,
            num_transformer_block=motion_num_layers,
            attention_block_types=("Temporal_Self", "Temporal_Self"),
            temporal_position_encoding=True,
            temporal_position_encoding_max_len=max_frames,
            temporal_attention_dim_div=1,
            zero_initialize=True,
        )
        self.motion_module.temporal_transformer.proj_out = zero_module(
            self.motion_module.temporal_transformer.proj_out
        )

    def forward(self, x, t, y):
        if x.dim() != 5:
            raise RuntimeError(f"expected video input (B,F,C,H,W), got {x.shape}")
        B, F_frames, _, _, _ = x.shape
        if F_frames > self.max_frames:
            raise RuntimeError(f"frames={F_frames} > max_frames={self.max_frames}")
        x = rearrange(x, "b f c h w -> (b f) c h w")
        t = t.reshape(-1)
        if t.shape[0] == B:
            t = repeat(t, "b -> (b f)", f=F_frames)
        y = y.reshape(-1)
        if y.shape[0] == B:
            y = repeat(y, "b -> (b f)", f=F_frames)
        c = self.t_embedder(t) + self.y_embedder(y)
        x = self.forward_tokens(x, c)
        patches = x.shape[1]
        hp = int(patches**0.5)
        if hp * hp != patches:
            raise RuntimeError(f"Patch token count {patches} is not square")
        x = rearrange(x, "(b f) (h w) d -> b f d h w", b=B, f=F_frames, h=hp, w=hp)
        x = self.motion_module(x)
        x = rearrange(x, "b f d h w -> (b f) (h w) d")
        x = self.final_layer(x, c)
        return self._unpatchify(x, B=B, F_frames=F_frames)
