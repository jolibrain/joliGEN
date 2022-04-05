import torch
from torch import nn as nn
from ..utils import init_net
import numpy as np
import torch.nn.functional as F


class SpatialCorrelativeLoss(nn.Module):
    """
    learnable patch-based spatially-correlative loss with contrastive learning
    """

    def __init__(
        self,
        loss_mode="cos",
        patch_nums=256,
        patch_size=32,
        norm=True,
        use_conv=True,
        init_type="normal",
        init_gain=0.02,
        gpu_ids=[],
        T=0.1,
    ):
        super(SpatialCorrelativeLoss, self).__init__()
        self.patch_sim = PatchSim(
            patch_nums=patch_nums, patch_size=patch_size, norm=norm
        )
        self.patch_size = patch_size
        self.patch_nums = patch_nums
        self.norm = norm
        self.use_conv = use_conv
        self.conv_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.loss_mode = loss_mode
        self.T = T
        self.criterion = nn.L1Loss() if norm else nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def update_init_(self):
        self.conv_init = True

    def create_conv(self, feat, layer):
        """
        create the 1*1 conv filter to select the features for a specific task
        :param feat: extracted features from a pretrained VGG or encoder for the similarity and dissimilarity map
        :param layer: different layers use different filter
        :return:
        """
        input_nc = feat.size(1)
        output_nc = max(32, input_nc // 4)
        conv = nn.Sequential(
            *[
                nn.Conv2d(input_nc, output_nc, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(output_nc, output_nc, kernel_size=1),
            ]
        )
        conv.to(feat.device)
        setattr(self, "conv_%d" % layer, conv)
        init_net(conv, self.init_type, self.init_gain)

    def cal_sim(self, f_src, f_tgt, f_other=None, layer=0, patch_ids=None):
        """
        calculate the similarity map using the fixed/learned query and key
        :param f_src: feature map from source domain
        :param f_tgt: feature map from target domain
        :param f_other: feature map from other image (only used for contrastive learning for spatial network)
        :return:
        """
        if self.use_conv:
            if not self.conv_init:
                self.create_conv(f_src, layer)
            conv = getattr(self, "conv_%d" % layer)
            f_src, f_tgt = conv(f_src), conv(f_tgt)
            f_other = conv(f_other) if f_other is not None else None
        sim_src, patch_ids = self.patch_sim(f_src, patch_ids)
        sim_tgt, patch_ids = self.patch_sim(f_tgt, patch_ids)
        if f_other is not None:
            sim_other, _ = self.patch_sim(f_other, patch_ids)
        else:
            sim_other = None

        return sim_src, sim_tgt, sim_other

    def compare_sim(self, sim_src, sim_tgt, sim_other):
        """
        measure the shape distance between the same shape and different inputs
        :param sim_src: the shape similarity map from source input image
        :param sim_tgt: the shape similarity map from target output image
        :param sim_other: the shape similarity map from other input image
        :return:
        """
        B, Num, N = sim_src.size()
        if self.loss_mode == "info" or sim_other is not None:
            sim_src = F.normalize(sim_src, dim=-1)
            sim_tgt = F.normalize(sim_tgt, dim=-1)
            sim_other = F.normalize(sim_other, dim=-1)
            sam_neg1 = (sim_src.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_neg2 = (sim_tgt.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = (sim_src.bmm(sim_tgt.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = torch.cat([sam_self, sam_neg1, sam_neg2], dim=-1)
            loss = self.cross_entropy_loss(
                sam_self,
                torch.arange(
                    0, sam_self.size(0), dtype=torch.long, device=sim_src.device
                )
                % (Num),
            )
        else:
            tgt_sorted, _ = sim_tgt.sort(dim=-1, descending=True)
            num = int(N / 4)
            src = torch.where(
                sim_tgt < tgt_sorted[:, :, num : num + 1], 0 * sim_src, sim_src
            )
            tgt = torch.where(
                sim_tgt < tgt_sorted[:, :, num : num + 1], 0 * sim_tgt, sim_tgt
            )
            if self.loss_mode == "l1":
                loss = self.criterion((N / num) * src, (N / num) * tgt)
            elif self.loss_mode == "cos":
                sim_pos = F.cosine_similarity(src, tgt, dim=-1)
                loss = self.criterion(torch.ones_like(sim_pos), sim_pos)
            else:
                raise NotImplementedError(
                    "padding [%s] is not implemented" % self.loss_mode
                )

        return loss

    def loss(self, f_src, f_tgt, f_other=None, layer=0):
        """
        calculate the spatial similarity and dissimilarity loss for given features from source and target domain
        :param f_src: source domain features
        :param f_tgt: target domain features
        :param f_other: other random sampled features
        :param layer:
        :return:
        """
        sim_src, sim_tgt, sim_other = self.cal_sim(f_src, f_tgt, f_other, layer)
        # if sim_other is not None:
        #     loss=self.barlow_loss_func(sim_src, sim_tgt)+self.barlow_loss_func(sim_other, sim_tgt)
        # else:
        #     loss=self.barlow_loss_func(sim_src, sim_tgt)
        # calculate the spatial similarity for source and target domain
        loss = self.compare_sim(sim_src, sim_tgt, sim_other)
        return loss

    def barlow_loss_func(self, z1, z2, lamb=5e-3, scale_loss=0.025):
        ### data preprocess
        # print(z1.shape)
        _, N, D = z1.size()
        z1 = z1.reshape(N, D)
        z2 = z2.reshape(N, D)
        # print(z1.shape)

        # to match the original code
        bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
        z1 = bn(z1)
        z2 = bn(z2)

        corr = torch.einsum("bi, bj -> ij", z1, z2) / N

        diag = torch.eye(D, device=corr.device)
        cdif = (corr - diag).pow(2)
        cdif[~diag.bool()] *= lamb
        loss = scale_loss * cdif.sum()
        return loss


class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class PatchSim(nn.Module):
    """Calculate the similarity in selected patches"""

    def __init__(self, patch_nums=256, patch_size=None, norm=True):
        super(PatchSim, self).__init__()
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.use_norm = norm

    def forward(self, feat, patch_ids=None):
        """
        Calculate the similarity for selected patches
        """
        B, C, W, H = feat.size()
        feat = feat - feat.mean(dim=[-2, -1], keepdim=True)
        feat = F.normalize(feat, dim=1) if self.use_norm else feat / np.sqrt(C)
        query, key, patch_ids = self.select_patch(feat, patch_ids=patch_ids)
        patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key) / 10)
        if patch_ids is not None:
            patch_sim = patch_sim.view(B, len(patch_ids), -1)

        return patch_sim, patch_ids

    def select_patch(self, feat, patch_ids=None):
        """
        Select the patches
        """
        B, C, W, H = feat.size()
        pw, ph = self.patch_size, self.patch_size
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # B*N*C N=W*H
        if self.patch_nums > 0:
            if patch_ids is None:
                patch_ids = torch.randperm(feat_reshape.size(1), device=feat.device)
                patch_ids = patch_ids[: int(min(self.patch_nums, patch_ids.size(0)))]
            feat_query = feat_reshape[:, patch_ids, :]  # B*Num*C
            feat_key = []
            Num = feat_query.size(1)
            if pw < W and ph < H:
                # pos_x, pos_y = patch_ids // W, patch_ids % W
                pos_x, pos_y = (
                    torch.div(patch_ids, W, rounding_mode="floor"),
                    patch_ids % W,
                )
                # patch should in the feature
                left, top = pos_x - int(pw / 2), pos_y - int(ph / 2)
                left, top = (
                    torch.where(left > 0, left, torch.zeros_like(left)),
                    torch.where(top > 0, top, torch.zeros_like(top)),
                )
                start_x = torch.where(
                    left > (W - pw), (W - pw) * torch.ones_like(left), left
                )
                start_y = torch.where(
                    top > (H - ph), (H - ph) * torch.ones_like(top), top
                )
                for i in range(Num):
                    feat_key.append(
                        feat[
                            :,
                            :,
                            start_x[i] : start_x[i] + pw,
                            start_y[i] : start_y[i] + ph,
                        ]
                    )  # B*C*patch_w*patch_h
                feat_key = torch.stack(feat_key, dim=0).permute(
                    1, 0, 2, 3, 4
                )  # B*Num*C*patch_w*patch_h
                feat_key = feat_key.reshape(B * Num, C, pw * ph)  # Num * C * N
                feat_query = feat_query.reshape(B * Num, 1, C)  # Num * 1 * C
            else:  # if patch larger than features size, use B * C * N (H * W)
                feat_key = feat.reshape(B, C, W * H)
        else:
            feat_query = feat.reshape(B, C, H * W).permute(0, 2, 1)  # B * N (H * W) * C
            feat_key = feat.reshape(B, C, H * W)  # B * C * N (H * W)

        return feat_query, feat_key, patch_ids


class SpatialLoss:
    def __init__(self, criterion, attn_layers, norm):
        self.criterion = criterion
        self.attn_layers = attn_layers
        self.n_layers = len(attn_layers)
        self.norm = norm

    def norm_img(self, img):  # dont forget to detach if needed
        norm_img = self.norm((img + 1)) * 0.5
        return norm_img

    def prepare_input(self, imgs):
        return_imgs = []
        for img in imgs:
            if img is not None:
                norm = self.norm_img(
                    img
                )  # norm_fake_idt_B = self.normalization((self.idt_B + 1) * 0.5)
            else:
                norm = None
            return_imgs.append(norm)
        return return_imgs

    def __call__(self, net, mask, src, tgt, other=None):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        src, tgt, other = self.prepare_input([src, tgt, other])

        if mask is not None:
            src = src * mask
            tgt = tgt * mask
        # feats_src = net(src, self.attn_layers, encode_only=True)  #
        feats_src = net.get_feats(src, self.attn_layers)
        # feats_tgt = net(tgt, self.attn_layers, encode_only=True)  #
        feats_tgt = net.get_feats(tgt, self.attn_layers)
        if other is not None:
            # feats_oth = net(torch.flip(other, [2, 3]), self.attn_layers, encode_only=True)
            feats_oth = net.get_feats(torch.flip(other, [2, 3]), self.attn_layers)
        else:
            feats_oth = [None for _ in range(self.n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(
            zip(feats_src, feats_tgt, feats_oth)
        ):

            loss = self.criterion.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.criterion.conv_init:
            self.criterion.update_init_()

        return total_loss / self.n_layers
