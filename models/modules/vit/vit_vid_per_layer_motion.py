# --------------------------------------------------------
# JiTVideo (pretrained JiT backbone) + MotionModule every N layers
# --------------------------------------------------------
# - Input  : x (B, F, C, H, W)
# - Output : out (B, F, C, H, W)
# --------------------------------------------------------
import math

import torch.nn as nn

from .vit_motion import MotionModule
from .vit_video_base import BaseJiTVideo, JiTVid_VARIANT_CONFIGS


class JiTViD(BaseJiTVideo):
    """
    Video Transformer with motion blocks inserted throughout the JiT stack.
    """

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
        motion_num_heads=8,
        motion_num_layers=2,
        motion_every=1,
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
            max_frames=max_frames,
            patch_stride_divisor=patch_stride_divisor,
        )
        if motion_every < 1:
            raise ValueError("motion_every must be >= 1")
        self.motion_every = motion_every

        num_motion_points = math.ceil(depth / self.motion_every)
        self.motion_modules = nn.ModuleList(
            [
                MotionModule(
                    in_channels=hidden_size,
                    num_attention_heads=motion_num_heads,
                    num_transformer_block=motion_num_layers,
                    attention_block_types=("Temporal_Self", "Temporal_Self"),
                    temporal_position_encoding=True,
                    temporal_position_encoding_max_len=max_frames,
                    temporal_attention_dim_div=1,
                    zero_initialize=True,
                )
                for _ in range(num_motion_points)
            ]
        )

        self.initialize_weights()

    def forward(self, x, t, y):
        """
        x: (B, F, C, H, W) tensor of video inputs
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """

        batch_size, num_frames, x, c, y_emb = self._embed_video_inputs(x, t, y)

        in_context_inserted = False
        for block_index in range(len(self.blocks)):
            x, in_context_inserted = self._run_block(
                x, c, y_emb, block_index, in_context_inserted
            )

            if ((block_index + 1) % self.motion_every) == 0:
                motion_index = block_index // self.motion_every
                prefix_tokens = self.in_context_len if in_context_inserted else 0
                x = self._apply_motion_to_patch_tokens(
                    x,
                    batch_size,
                    num_frames,
                    self.motion_modules[motion_index],
                    prefix_tokens=prefix_tokens,
                )

        x = self._strip_in_context_tokens(x, in_context_inserted)
        output = self._decode_tokens(x, c)
        return self._restore_video_output(output, batch_size, num_frames)
