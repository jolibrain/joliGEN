import torch
from einops import rearrange

from .vit import BaseJiT, JiT_VARIANT_CONFIGS


JiTVid_VARIANT_CONFIGS = {
    variant.replace("JiT-", "JiTVid-"): dict(config)
    for variant, config in JiT_VARIANT_CONFIGS.items()
    if variant != "JiT-B/8"
}


class BaseJiTVideo(BaseJiT):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=768,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        cond_embed_dim=None,
        max_frames=8,
        patch_stride_divisor=1,
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            num_classes=num_classes,
            bottleneck_dim=bottleneck_dim,
            in_context_len=in_context_len,
            in_context_start=in_context_start,
            cond_embed_dim=cond_embed_dim,
            patch_stride_divisor=patch_stride_divisor,
        )
        self.max_frames = max_frames

    def _expand_condition_per_frame(self, values, batch_size, num_frames, name):
        values = values.reshape(-1)
        if values.shape[0] == batch_size:
            return values[:, None].expand(-1, num_frames).reshape(
                batch_size * num_frames
            )
        if values.shape[0] == batch_size * num_frames:
            return values
        raise RuntimeError(
            f"Unexpected {name} length {values.shape[0]} "
            f"(expected {batch_size} or {batch_size * num_frames})."
        )

    def _embed_video_inputs(self, x, t, y):
        assert x.dim() == 5, f"expected video input (B,F,C,H,W), got {x.shape}"
        batch_size, num_frames, _, _, _ = x.shape
        assert (
            num_frames <= self.max_frames
        ), f"frames={num_frames} > max_frames={self.max_frames}"

        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self._embed_patches(x)

        t2d = self._expand_condition_per_frame(
            t, batch_size, num_frames, "timestep"
        )
        y2d = self._expand_condition_per_frame(y, batch_size, num_frames, "label")
        c, y_emb = self._embed_condition(t2d, y2d)
        return batch_size, num_frames, x, c, y_emb

    def _apply_motion_to_patch_tokens(
        self, x, batch_size, num_frames, motion_module, prefix_tokens=0
    ):
        if prefix_tokens > 0:
            prefix = x[:, :prefix_tokens]
            patches = x[:, prefix_tokens:]
        else:
            prefix = None
            patches = x

        height, width = self.x_embedder.grid_size
        patches = rearrange(
            patches,
            "(b f) (h w) d -> b f d h w",
            b=batch_size,
            f=num_frames,
            h=height,
            w=width,
        )
        patches = motion_module(patches)
        patches = rearrange(patches, "b f d h w -> (b f) (h w) d")

        if prefix is None:
            return patches
        return torch.cat([prefix, patches], dim=1)

    def _restore_video_output(self, output, batch_size, num_frames):
        return rearrange(
            output, "(b f) c h w -> b f c h w", b=batch_size, f=num_frames
        )
