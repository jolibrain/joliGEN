import torch
import torch.nn as nn
import numpy as np

# from timm import create_model
from timm.models.vision_transformer import PatchEmbed, Block

from .projected_d.projector import _make_projector


class VitEncoder(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        num_heads,
        mlp_ratio,
        norm_layer,
        depth,
    ):
        super(VitEncoder, self).__init__()

        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def forward(self, input):
        """Standard forward"""
        output, _ = self.compute_feats(input)
        output = self.norm(output)

        return output

    def get_feats(self, input, extract_layer_ids):
        _, feats = self.compute_feats(input, extract_layer_ids)
        return feats

    def compute_feats(self, input, extract_layer_ids=[]):
        if -1 in extract_layer_ids:
            extract_layer_ids.append(len(self.encoder))
        feat = input

        # embed patches
        feat = self.patch_embed(feat)

        # add pos embed w/o cls token
        feat = feat + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(feat.shape[0], -1, -1)

        feat = torch.cat((cls_tokens, feat), dim=1)

        # apply Transformer blocks
        feats = []
        for layer_id, blk in enumerate(self.blocks):
            feat = blk(feat)
            if layer_id in extract_layer_ids:
                feats.append(feat)

        return feat, feats  # return both output and intermediate features

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class VitDecoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        decoder_embed_dim,
        decoder_num_heads,
        mlp_ratio,
        norm_layer,
        num_patches,
        decoder_depth,
        patch_size,
        in_chans,
    ):
        super(VitDecoder, self).__init__()

        self.num_patches = num_patches

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class VitGenerator(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        encoder_architecture,
        embed_dim=400,  # 768,
        depth=8,  # ,12,
        num_heads=8,  # 12,
        decoder_embed_dim=400,  # 512,
        decoder_depth=4,  # 8,
        decoder_num_heads=8,  # 16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super(VitGenerator, self).__init__()

        self.patch_size = patch_size

        if encoder_architecture == "custom":
            self.encoder = VitEncoder(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                depth=depth,
            )

        else:
            self.encoder, _ = _make_projector(
                projector_model=encoder_architecture,
                cout=None,
                proj_type=0,
                expand=None,
                config_path=None,
                weight_path=None,
                interp=img_size,
            )

            embed_dim = self.encoder.CHANNELS[0]

            configure_get_feats_vit_timm(self.encoder)
            configure_compute_feats_vit_timm(self.encoder)
            configure_forward_vit_timm(self.encoder)

        self.decoder = VitDecoder(
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            num_patches=self.encoder.patch_embed.num_patches,
            decoder_depth=decoder_depth,
            patch_size=patch_size,
            in_chans=in_chans,
        )

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """

        print(x.shape)

        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)

        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def unpatchify_feats(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """

        # Remove cls token
        print("input un patch feat", x.shape)
        # x = x[:, 1:, :]

        p = int(x.shape[2] ** 0.5)  # self.patch_size
        print("p", p)
        h = w = int(x.shape[1] ** 0.5)
        print(x.shape)
        print(h, x.shape[1])
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p * p))
        print("x", x.shape)
        x = torch.einsum("nhwp->nphw", x)
        print("x ein", x.shape)
        imgs = x  # .reshape(shape=(x.shape[0], p, p, h, w))
        print(imgs.shape)
        return imgs

    def get_feats(self, input, extract_layer_ids):
        feats = self.encoder.get_feats(input, extract_layer_ids)
        return_feats = []
        for feat in feats:
            return_feats.append(self.unpatchify_feats(feat))
        return return_feats

    def forward(self, x):
        print(x.shape)
        x = self.encoder(x)
        print("after encoder", x.shape)
        x = self.decoder(x)
        print(x.shape)
        return self.unpatchify(x)


###############################


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def configure_get_feats_vit_timm(net):
    def get_feats(input, extract_layer_ids):
        _, feats = net.compute_feats(input, extract_layer_ids)
        return feats

    net.get_feats = get_feats


def configure_forward_vit_timm(net):
    def forward(input):
        output, _ = net.compute_feats(input)
        return output

    net.forward = forward


def configure_compute_feats_vit_timm(net):
    def compute_feats(x, extract_layer_ids=[]):
        x = net.patch_embed(x)
        x = torch.cat((net.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = net.pos_drop(x + net.pos_embed)
        feats = []
        feat = x
        for i, block in enumerate(net.blocks):
            feat = block(feat)
            if i in extract_layer_ids:
                feats.append(feat.transpose(2, 1).contiguous())
        feats.append(feat.transpose(2, 1).contiguous())

        return x, feats

    net.compute_feats = compute_feats
