import torch.nn as nn
from .utils import init_net
import torch


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type="normal", init_gain=0.02, nc=256):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain

    def set_device(self, device):
        self.device = device

    def data_dependent_initialize(self, feats):
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)]
            )
            setattr(self, "mlp_%d" % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain)
        for mlp_id in range(len(feats)):
            setattr(
                self,
                "mlp_%d" % mlp_id,
                getattr(self, "mlp_%d" % mlp_id).to(self.device),
            )
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []

        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id].squeeze()
                else:
                    patch_id = torch.randperm(
                        feat_reshape.shape[1], device=feats[0].device
                    )
                    patch_id = patch_id[
                        : int(min(num_patches, patch_id.shape[0]))
                    ]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(
                    0, 1
                )  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, "mlp_%d" % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id.unsqueeze(0))
            x_sample = torch.nn.functional.normalize(x_sample, eps=1e-7)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape(
                    [B, x_sample.shape[-1], H, W]
                )
            return_feats.append(x_sample)
        return return_feats, return_ids


class PatchSampleF_QSAttn(nn.Module):
    def __init__(
        self, use_mlp=False, init_type="normal", init_gain=0.02, nc=256, gpu_ids=[]
    ):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF_QSAttn, self).__init__()
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def set_device(self, device):
        self.device = device

    def data_dependent_initialize(self, feats):
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)]
            )
            mlp.cuda()
            setattr(self, "mlp_%d" % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain)
        for mlp_id in range(len(feats)):
            setattr(
                self,
                "mlp_%d" % mlp_id,
                getattr(self, "mlp_%d" % mlp_id).to(self.device),
            )
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None, attn_mats=None):
        return_ids = []
        return_feats = []
        return_mats = []
        k_s = 7  # kernel size in unfold
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # B*HW*C
            if num_patches > 0:
                if feat_id < 3:
                    if patch_ids is not None:
                        patch_id = patch_ids[feat_id]
                    else:
                        patch_id = torch.randperm(
                            feat_reshape.shape[1], device=feats[0].device
                        )  # random id in [0, HW]
                        patch_id = patch_id[
                            : int(min(num_patches, patch_id.shape[0]))
                        ]  # .to(patch_ids.device)
                    x_sample = feat_reshape[:, patch_id, :].flatten(
                        0, 1
                    )  # reshape(-1, x.shape[1])
                    attn_qs = torch.zeros(1).to(feat.device)
                else:
                    if attn_mats is not None:
                        attn_qs = attn_mats[feat_id]
                    else:
                        feat_local = F.unfold(
                            feat, kernel_size=k_s, stride=1, padding=3
                        )  # (B, ks*ks*C, L)
                        L = feat_local.shape[2]
                        feat_k_local = (
                            feat_local.permute(0, 2, 1)
                            .reshape(B, L, k_s * k_s, C)
                            .flatten(0, 1)
                        )  # (B*L, ks*ks, C)
                        feat_q_local = feat_reshape.reshape(B * L, C, 1)
                        dots_local = torch.bmm(
                            feat_k_local, feat_q_local
                        )  # (B*L, ks*ks, 1)
                        attn_local = dots_local.softmax(dim=1)
                        attn_local = attn_local.reshape(B, L, -1)  # (B, L, ks*ks)
                        prob = -torch.log(attn_local)
                        prob = torch.where(
                            torch.isinf(prob), torch.full_like(prob, 0), prob
                        )
                        entropy = torch.sum(torch.mul(attn_local, prob), dim=2)
                        _, index = torch.sort(entropy)
                        patch_id = index[:, :num_patches]
                        feat_q_global = feat_reshape
                        feat_k_global = feat_reshape.permute(0, 2, 1)
                        dots_global = torch.bmm(
                            feat_q_global, feat_k_global
                        )  # (B, HW, HW)
                        attn_global = dots_global.softmax(dim=2)
                        attn_qs = attn_global[torch.arange(B)[:, None], patch_id, :]
                    feat_reshape = torch.bmm(attn_qs, feat_reshape)  # (B, n_p, C)
                    x_sample = feat_reshape.flatten(0, 1)
                    patch_id = []
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, "mlp_%d" % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            return_mats.append(attn_qs)
            x_sample = torch.nn.functional.normalize(x_sample, eps=1e-7)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape(
                    [B, x_sample.shape[-1], H, W]
                )
            return_feats.append(x_sample)
        return return_feats, return_ids, return_mats
