import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .fair_vicregl.utils import gather_center, off_diagonal


class VICRegFairLoss(nn.Module):
    def __init__(
        self,
        inv_coeff: float = 25.0,
        var_coeff: float = 15.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
    ):

        super().__init__()
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.alg_cut_nce_includes_all_negatives_from_minibatch = False

    def global_loss(self, embedding):
        num_views = len(embedding)
        inv_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                print(i, j)
                inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
                iter_ = iter_ + 1
        inv_loss = self.inv_coeff * inv_loss / iter_

        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(num_views):
            x = gather_center(embedding[i])
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + off_diagonal(cov_x).pow_(2).sum().div(
                self.embedding_dim
            )
            iter_ = iter_ + 1
        var_loss = self.var_coeff * var_loss / iter_
        cov_loss = self.cov_coeff * cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def forward(
        self, feat_q: torch.Tensor, feat_k: torch.Tensor, current_batch, weight
    ):

        dim = feat_q.shape[1]

        if self.alg_cut_nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            self.batch_dim_for_bmm = 1
        else:
            self.batch_dim_for_bmm = current_batch

        feat_q = feat_q.view(self.batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(self.batch_dim_for_bmm, -1, dim)

        embedding = [feat_q, feat_k]

        return self.global_loss(embedding)
