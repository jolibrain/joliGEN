from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
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

    def representation_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the representation loss.
        Force the representations of the same object to be similar.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].
            y: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The representation loss.
                Shape of [1,].
        """
        return F.mse_loss(x, y)

    @staticmethod
    def variance_loss(x: torch.Tensor, gamma: float) -> torch.Tensor:
        """Computes the local variance loss.
        This is slightly different than the global variance loss because the
        variance is computed per patch.
        ---
        Args:
            x: Features map.
                Shape of [batch_size, n_patches, representation_size].
        ---
        Returns:
            The variance loss.
                Shape of [1,].
        """

        x = x - x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1) + 0.0001)
        # std = x.std(dim=1)

        var_loss = F.relu(gamma - std).mean()
        return var_loss

    @staticmethod
    def covariance_loss(x: torch.Tensor) -> torch.Tensor:
        """Computes the local covariance loss.
        This is slightly different than the global covariance loss because the
        covariance is computed per patch.
        ---
        Args:
            x: Features map.
                Shape of [batch_size, n_patches, representation_size].
        ---
        Returns:
            The covariance loss.
                Shape of [1,].
        """
        x = x - x.mean(dim=1, keepdim=True)
        x_T = x.transpose(1, 2)
        cov = (x_T @ x) / (x.shape[1] - 1)

        non_diag = ~torch.eye(x.shape[2], device=x.device, dtype=torch.bool)
        cov_loss = cov[..., non_diag].pow(2).sum() / (x.shape[2] * x.shape[0])
        return cov_loss

    # def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
    def forward(
        self, feat_q: torch.Tensor, feat_k: torch.Tensor, current_batch, weight
    ) -> Dict[str, torch.Tensor]:
        """Computes the VICReg loss.

        ---
        Args:
            x: Features map.
                Shape of [batch_size, representation_size].
            y: Features map.
                Shape of [batch_size, representation_size].

        ---
        Returns:
            The VICReg loss.
                Dictionary where values are of shape of [1,].
        """

        dim = feat_q.shape[1]

        if self.alg_cut_nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            self.batch_dim_for_bmm = 1
        else:
            self.batch_dim_for_bmm = current_batch

        feat_q = feat_q.view(self.batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(self.batch_dim_for_bmm, -1, dim)

        x = feat_q
        y = feat_k

        metrics = dict()
        metrics["inv-loss"] = self.inv_coeff * self.representation_loss(x, y)
        metrics["var-loss"] = (
            self.var_coeff
            * (self.variance_loss(x, self.gamma) + self.variance_loss(y, self.gamma))
            / 2
        )
        metrics["cov-loss"] = (
            self.cov_coeff * (self.covariance_loss(x) + self.covariance_loss(y)) / 2
        )
        metrics["loss"] = sum(metrics.values())

        return metrics
