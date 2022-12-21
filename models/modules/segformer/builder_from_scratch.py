from torch import nn


from models.modules.segformer.segformer_head import SegformerHead
from models.modules.segformer.backbone import MixVisionTransformer


class JoliSegformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.with_neck = False

        cfg_file = cfg

        cfg_backbone = cfg_file["backbone"]
        self.backbone = MixVisionTransformer(**cfg_backbone)

        cfg_decode_head = cfg_file["decode_head"]
        self.decode_head = SegformerHead(**cfg_decode_head)

        if "auxiliary_head" in cfg_file.keys():
            cfg_auxiliary_head = cfg_file["auxiliary_head"]
            self.auxiliary_head = SegformerHead(**cfg_auxiliary_head)

    def decode_head_forward(self, x):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward(x)
        return seg_logits

    def auxiliary_head_forward(self, x):
        """Run forward function and calculate loss for auxiliary head in
        inference."""
        seg_logits = self.auxiliary_head.forward(x)
        return seg_logits

    def extract_feat(self, img, extract_layer_ids=[]):
        """Extract features from images."""
        x = self.backbone.compute_feat(img, extract_layer_ids)
        x, feats = x
        if self.with_neck:
            x = self.neck(x)
        return x, feats
