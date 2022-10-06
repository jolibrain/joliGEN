from packaging import version
import torch
from torch import nn
import math
from .sinkhorn import OT
import numpy as np
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm + 1e-7)
        return out


class MoNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = (
            torch.uint8
            if version.parse(torch.__version__) < version.parse("1.2.0")
            else torch.bool
        )
        self.l2_norm = Normalize(2)

    def forward(self, feat_q, feat_k, current_batch):
        eps = 1.0  # default
        cost_type = "hard"  # default for cut
        neg_term_weight = 1.0  # default

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.alg_cut_nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = current_batch

        # if self.loss_type == 'MoNCE':
        ot_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        ot_k = feat_k.view(batch_dim_for_bmm, -1, dim).detach()
        f = OT(ot_q, ot_k, eps=eps, max_iter=50, cost_type=cost_type)
        f = (
            f.permute(0, 2, 1) * (self.opt.alg_cut_num_patches - 1) * neg_term_weight
            + 1e-8
        )

        feat_k = feat_k.detach()
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        # if self.loss_type == 'MoNCE':
        l_neg_curbatch = l_neg_curbatch + torch.log(f) * self.opt.alg_cut_nce_T

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[
            None, :, :
        ]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.alg_cut_nce_T
        loss = self.cross_entropy_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        )
        return loss
