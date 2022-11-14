from packaging import version
import torch
from torch import nn
import math
from .sinkhorn import OT
import numpy as np
import torch.nn.functional as F

from .base_NCE import BaseNCELoss


class MoNCELoss(BaseNCELoss):
    def __init__(self, opt):
        super().__init__(opt)

    def compute_l_neg_curbatch(self, feat_q, feat_k):
        eps = 1.0  # default
        cost_type = "hard"  # default for cut
        neg_term_weight = 1.0  # default

        ot_q = feat_q.view(self.batch_dim_for_bmm, -1, self.dim)
        ot_k = feat_k.view(self.batch_dim_for_bmm, -1, self.dim).detach()
        f = OT(ot_q, ot_k, eps=eps, max_iter=50, cost_type=cost_type)
        f = (
            f.permute(0, 2, 1) * (self.opt.alg_cut_num_patches - 1) * neg_term_weight
            + 1e-8
        )

        l_neg_curbatch, npatches = super().compute_l_neg_curbatch(feat_q, feat_k)

        l_neg_curbatch = l_neg_curbatch + torch.log(f) * self.opt.alg_cut_nce_T

        return l_neg_curbatch, npatches
