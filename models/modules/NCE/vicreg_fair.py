from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegFairLoss(nn.Module):
    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 15.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
    ):
        super().__init__()

        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

        self.gamma = gamma
        self.alg_cut_nce_includes_all_negatives_from_minibatch = False

    def forward(self, feat_q, feat_k, current_batch, weight):
        return self.forward_base(
            x=feat_q, y=feat_k, current_batch=current_batch, weight=weight
        )

    def forward_base(self, x, y, current_batch, weight):

        num_features = len(x)

        repr_loss = F.mse_loss(x, y)

        # repr_loss = F.mse_loss(torch.randn_like(x), torch.randn_like(y))

        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (current_batch - 1)
        cov_y = (y.T @ y) / (current_batch - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(
            cov_y
        ).pow_(2).sum().div(num_features)

        metrics = dict()
        metrics["inv-loss"] = self.sim_coeff * repr_loss
        metrics["var-loss"] = self.std_coeff * std_loss
        metrics["cov-loss"] = self.cov_coeff * cov_loss
        metrics["loss"] = sum(metrics.values())

        return metrics
