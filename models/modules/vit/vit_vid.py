# --------------------------------------------------------
# JiTVideo (pretrained JiT backbone) + MotionModule at LAST layer
# --------------------------------------------------------
# - Input  : x (B, F, C, H, W)
# - Output : out (B, F, C, H, W)
# --------------------------------------------------------
from .vit_motion import MotionModule
from .vit_video_base import BaseJiTVideo, JiTVid_VARIANT_CONFIGS


class JiTViD(BaseJiTVideo):
    """
    Just video Transformer.
    - Take video (B,F,C,H,W)
    - Run JiT blocks per-frame in batch flatten (B*F)
    - After last transformer block, apply MotionModule across time on TOKEN SPACE
    - Then run final_layer + unpatchify per frame and return (B,F,C,H,W)
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

        self.motion_module = MotionModule(
            in_channels=hidden_size,
            num_attention_heads=motion_num_heads,
            num_transformer_block=motion_num_layers,
            attention_block_types=("Temporal_Self", "Temporal_Self"),
            temporal_position_encoding=True,
            temporal_position_encoding_max_len=max_frames,
            temporal_attention_dim_div=1,
            zero_initialize=True,
        )

        self.initialize_weights()

    def forward(self, x, t, y):
        """
        x: (B, F, C, H, W) tensor of video inputs
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """

        batch_size, num_frames, x, c, y_emb = self._embed_video_inputs(x, t, y)
        x = self._run_blocks(x, c, y_emb)
        x = self._apply_motion_to_patch_tokens(
            x, batch_size, num_frames, self.motion_module
        )
        output = self._decode_tokens(x, c)
        return self._restore_video_output(output, batch_size, num_frames)
