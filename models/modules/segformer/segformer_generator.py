import os
import json

from torch import nn
from torch.nn.functional import interpolate as resize

from models.modules.resnet_architecture.resnet_generator import ResnetDecoder
from models.modules.attn_network import BaseGenerator_attn
from models.modules.segformer.config import load_config_file
from .builder_from_scratch import JoliSegformer


class SegformerBackbone(nn.Module):
    def __init__(
        self,
        jg_dir,
        G_config_segformer,
        input_nc,
        img_size,
        num_classes=10,
        final_conv=False,
        padding_type="zeros",
    ):
        super().__init__()

        cfg = load_config_file(os.path.join(jg_dir, G_config_segformer))

        cfg["decode_head"]["num_classes"] = num_classes
        cfg["backbone"]["in_channels"] = input_nc
        cfg["pretrained"] = None
        cfg["train_cfg"] = None
        cfg["auxiliary_head"] = cfg["decode_head"].copy()

        self.net = JoliSegformer(cfg)

        self.net.img_size = img_size

        self.use_final_conv = final_conv

        if self.use_final_conv:
            self.final_conv = ResnetDecoder(
                num_classes, input_nc, ngf=64, padding_type=padding_type
            )

    def compute_feats(self, input, extract_layer_ids=[]):
        outs, feats = self.net.extract_feat(input, extract_layer_ids)
        return outs, feats

    def forward(self, input):
        outs, _ = self.compute_feats(input)
        out = self.net.decode_head_forward(outs)
        if self.use_final_conv:
            out = self.final_conv(out)
        return out

    def get_feats(self, input, extract_layer_ids):
        _, feats = self.compute_feats(input, extract_layer_ids)
        return feats


class Segformer(SegformerBackbone):
    def __init__(
        self,
        jg_dir,
        G_config_segformer,
        input_nc,
        img_size,
        num_classes=10,
        final_conv=False,
        padding_type="zeros",
    ):
        super().__init__(
            jg_dir,
            G_config_segformer,
            input_nc,
            img_size,
            num_classes,
            final_conv,
            padding_type,
        )

    def forward(self, input):
        out = super().forward(input)

        out = resize(
            input=out,
            size=input.shape[-1],
            mode="bilinear",
        )
        return out


class SegformerGenerator_attn(BaseGenerator_attn):
    # initializers
    def __init__(
        self,
        jg_dir,
        G_config_segformer,
        input_nc,
        img_size,
        nb_mask_attn,
        nb_mask_input,
        final_conv=False,
        padding_type="zeros",
    ):  # nb_mask_attn : total number of attention masks, nb_mask_input :number of attention mask applied to input img directly
        super(SegformerGenerator_attn, self).__init__(nb_mask_attn, nb_mask_input)
        self.use_final_conv = final_conv
        self.tanh = nn.Tanh()
        self.input_nc = input_nc

        cfg = load_config_file(os.path.join(jg_dir, G_config_segformer))

        cfg["backbone"]["in_channels"] = self.input_nc
        cfg["pretrained"] = None
        cfg["train_cfg"] = None
        cfg["auxiliary_head"] = cfg["decode_head"].copy()

        if self.use_final_conv:
            num_cls = 256
        else:
            num_cls = self.input_nc * (self.nb_mask_attn - self.nb_mask_input)

        cfg["decode_head"]["num_classes"] = num_cls
        cfg["auxiliary_head"]["num_classes"] = self.nb_mask_attn

        self.segformer = JoliSegformer(cfg)

        self.segformer.train()
        self.softmax_ = nn.Softmax(dim=1)

        self.use_final_conv = final_conv

        if self.use_final_conv:
            self.final_conv = ResnetDecoder(
                num_cls,
                self.input_nc * (self.nb_mask_attn - self.nb_mask_input),
                ngf=64,
                padding_type=padding_type,
            )

    def compute_feats(self, input, extract_layer_ids=[]):
        outs, feats = self.segformer.extract_feat(input, extract_layer_ids)
        return outs, feats

    def compute_attention_content(self, outs):
        image = self.segformer.decode_head_forward(outs)
        if self.use_final_conv:
            image = self.final_conv(image)

        attention = self.segformer.auxiliary_head_forward(outs)
        images = []

        for i in range(self.nb_mask_attn - self.nb_mask_input):
            images.append(image[:, self.input_nc * i : self.input_nc * (i + 1), :, :])

        attention = self.softmax_(attention)
        attentions = []

        for i in range(self.nb_mask_attn):
            attentions.append(
                attention[:, i : i + 1, :, :].repeat(1, self.input_nc, 1, 1)
            )

        return attentions, images
