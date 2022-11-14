from packaging import version
import torch
from torch import nn

from .base_NCE import BaseNCELoss


class PatchHDCELoss(BaseNCELoss):
    def __init__(self, opt):
        super().__init__(opt)

    def forward(self, feat_q, feat_k, current_batch, weight):
        self.weight = weight
        return super().forward(feat_q, feat_k, current_batch)

    def compute_l_neg_curbatch(self, feat_q, feat_k):
        l_neg_curbatch, npatches = super().compute_l_neg_curbatch(feat_q, feat_k)
        # weighted by semantic relation
        if self.weight is not None:
            l_neg_curbatch *= self.weight
        return l_neg_curbatch, npatches

    def compute_loss(self, l_pos, l_neg):
        logits = (l_neg - l_pos) / self.opt.alg_cut_nce_T
        v = torch.logsumexp(logits, dim=1)
        loss_vec = torch.exp(v - v.detach())

        # for monitoring
        out_dummy = torch.cat((l_pos, l_neg), dim=1) / self.opt.alg_cut_nce_T

        CELoss_dummy = self.cross_entropy_loss(
            out_dummy,
            torch.zeros(out_dummy.size(0), dtype=torch.long, device=out_dummy.device),
        )

        loss = loss_vec - 1 + CELoss_dummy.detach()

        return loss
