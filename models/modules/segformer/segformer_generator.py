import os
from torch import nn

from models.modules.resnet_architecture.resnet_generator import ResnetDecoder
from models.modules.attn_network import BaseGenerator_attn
from .utils import configure_encoder_decoder, configure_mit


class Segformer(nn.Module):
    def __init__(
        self,
        jg_dir,
        G_config_segformer,
        img_size,
        num_classes=10,
        final_conv=False,
        padding_type="zeros",
    ):
        super().__init__()
        import mmcv

        cfg = mmcv.Config.fromfile(os.path.join(jg_dir, G_config_segformer))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.model.decode_head.num_classes = num_classes
        from mmseg.models import build_segmentor

        self.net = build_segmentor(
            cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg")
        )

        configure_encoder_decoder(self.net)
        self.net.img_size = img_size
        configure_mit(self.net.backbone)

        self.use_final_conv = final_conv

        if self.use_final_conv:
            self.final_conv = ResnetDecoder(
                num_classes, 3, ngf=64, padding_type=padding_type
            )

    def compute_feats(self, input, extract_layer_ids=[]):
        outs, feats = self.net.extract_feat(input, extract_layer_ids)
        return outs, feats

    def forward(self, input):
        outs, _ = self.compute_feats(input)
        out = self.net.decode(outs, use_resize=not self.use_final_conv)
        if self.use_final_conv:
            out = self.final_conv(out)
        return out

    def get_feats(self, input, extract_layer_ids):
        _, feats = self.compute_feats(input, extract_layer_ids)
        return feats


class SegformerGenerator_attn(BaseGenerator_attn):
    # initializers
    def __init__(
        self,
        jg_dir,
        G_config_segformer,
        img_size,
        nb_mask_attn,
        nb_mask_input,
        final_conv=False,
        padding_type="zeros",
    ):  # nb_mask_attn : total number of attention masks, nb_mask_input :number of attention mask applied to input img directly
        super(SegformerGenerator_attn, self).__init__(nb_mask_attn, nb_mask_input)
        self.use_final_conv = final_conv
        self.tanh = nn.Tanh()

        import mmcv

        cfg = mmcv.Config.fromfile(os.path.join(jg_dir, G_config_segformer))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.model.auxiliary_head = cfg.model.decode_head.copy()
        if self.use_final_conv:
            num_cls = 256
        else:
            num_cls = 3 * (self.nb_mask_attn - self.nb_mask_input)
        cfg.model.decode_head.num_classes = num_cls
        cfg.model.auxiliary_head.num_classes = self.nb_mask_attn
        from mmseg.models import build_segmentor

        self.segformer = build_segmentor(
            cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg")
        )
        self.segformer.train()
        self.softmax_ = nn.Softmax(dim=1)

        configure_encoder_decoder(self.segformer)
        self.segformer.img_size = img_size
        configure_mit(self.segformer.backbone)

        self.use_final_conv = final_conv

        if self.use_final_conv:
            self.final_conv = ResnetDecoder(
                num_cls,
                3 * (self.nb_mask_attn - self.nb_mask_input),
                ngf=64,
                padding_type=padding_type,
            )

    def compute_feats(self, input, extract_layer_ids=[]):
        outs, feats = self.segformer.extract_feat(input, extract_layer_ids)
        return outs, feats

    def compute_attention_content(self, outs):
        image = self.segformer.decode(outs, use_resize=not self.use_final_conv)
        if self.use_final_conv:
            image = self.final_conv(image)

        attention = self.segformer.decode_2(outs, use_resize=not self.use_final_conv)
        images = []

        for i in range(self.nb_mask_attn - self.nb_mask_input):
            images.append(image[:, 3 * i : 3 * (i + 1), :, :])

        attention = self.softmax_(attention)
        attentions = []

        for i in range(self.nb_mask_attn):
            attentions.append(attention[:, i : i + 1, :, :].repeat(1, 3, 1, 1))

        return attentions, images
