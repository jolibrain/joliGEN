import torch
from torch import nn
from packaging import version


class BaseNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = (
            torch.uint8
            if version.parse(torch.__version__) < version.parse("1.2.0")
            else torch.bool
        )

    def forward(self, feat_q, feat_k, current_batch, **unused_args):
        self.dim = feat_q.shape[1]

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.alg_cut_nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            self.batch_dim_for_bmm = 1
        else:
            self.batch_dim_for_bmm = current_batch

        # Positive logits
        l_pos = self.compute_pos_logit(feat_q, feat_k)

        # neg logit
        l_neg = self.compute_neg_logit(feat_q, feat_k)

        loss = self.compute_loss(l_pos, l_neg)

        return loss

    def compute_loss(self, l_pos, l_neg):
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.alg_cut_nce_T

        loss = self.cross_entropy_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=out.device)
        )
        return loss

    def compute_pos_logit(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        return l_pos

    def compute_l_neg_curbatch(self, feat_q, feat_k):
        """Returns negative examples"""
        dim = feat_q.shape[1]
        # reshape features to batch size
        feat_q = feat_q.view(self.batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(self.batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1).contiguous())
        return l_neg_curbatch, npatches

    def compute_neg_logit(self, feat_q, feat_k):
        l_neg_curbatch, npatches = self.compute_l_neg_curbatch(feat_q, feat_k)
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[
            None, :, :
        ]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)
        return l_neg
