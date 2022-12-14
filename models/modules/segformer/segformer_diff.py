import os
from torch import nn
import torch

from models.modules.resnet_architecture.resnet_generator import ResnetDecoder
from models.modules.attn_network import BaseGenerator_attn
from .utils import configure_encoder_decoder, configure_mit
from models.modules.diffusion_utils import gamma_embedding

from mmseg.ops import resize


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
        import mmcv

        cfg = mmcv.Config.fromfile(os.path.join(jg_dir, G_config_segformer))
        cfg.model.backbone.in_channels = input_nc
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
        out = self.net.decode(outs)
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


class SegformerGeneratorDiff_attn(BaseGenerator_attn):
    # initializers
    def __init__(
        self,
        jg_dir,
        G_config_segformer,
        input_nc,
        img_size,
        nb_mask_attn,
        nb_mask_input,
        inner_channel,
        n_timestep_train,
        n_timestep_test,
        final_conv=False,
        padding_type="zeros",
    ):  # nb_mask_attn : total number of attention masks, nb_mask_input :number of attention mask applied to input img directly
        super(SegformerGeneratorDiff_attn, self).__init__(nb_mask_attn, nb_mask_input)
        self.use_final_conv = final_conv
        self.tanh = nn.Tanh()

        import mmcv

        cfg = mmcv.Config.fromfile(os.path.join(jg_dir, G_config_segformer))
        cfg.model.backbone.in_channels = input_nc + 1
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

        self.beta_schedule = {
            "train": {
                "schedule": "linear",
                "n_timestep": n_timestep_train,
                "linear_start": 1e-6,
                "linear_end": 0.01,
            },
            "test": {
                "schedule": "linear",
                "n_timestep": n_timestep_test,
                "linear_start": 1e-4,
                "linear_end": 0.09,
            },
        }

        cond_embed_dim = inner_channel  # *4

        self.inner_channel = inner_channel
        self.img_size = img_size

        self.cond_embed = nn.Sequential(
            nn.Linear(inner_channel, cond_embed_dim),
            torch.nn.SiLU(),
            nn.Linear(cond_embed_dim, img_size * img_size),
        )

    def compute_feats(self, input, gammas, extract_layer_ids=[]):
        if gammas is None:
            b = input.shape[0]
            gammas = torch.ones((b,)).to(input.device)

        gammas = gammas.view(-1)

        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))

        emb = emb.view(input.shape[0], 1, self.img_size, self.img_size)

        input = torch.cat([input, emb], 1)

        outs, feats = self.segformer.extract_feat(input, extract_layer_ids)
        return outs, feats

    def get_feats(self, input, gammas=None, extract_layer_ids=[]):
        if gammas is None:
            b = input.shape[0]
            gammas = torch.ones((b,)).to(input.device)
        _, feats = self.compute_feats(
            input, gammas=gammas, extract_layer_ids=extract_layer_ids
        )
        return feats

    def extract(self, a, t, x_shape=(1, 1, 1, 1)):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def compute_attention_content(self, outs):
        image = self.segformer.decode(outs)
        if self.use_final_conv:
            image = self.final_conv(image)

        attention = self.segformer.decode_2(outs)
        images = []

        for i in range(self.nb_mask_attn - self.nb_mask_input):
            images.append(image[:, 3 * i : 3 * (i + 1), :, :])

        attention = self.softmax_(attention)
        attentions = []

        for i in range(self.nb_mask_attn):
            attentions.append(attention[:, i : i + 1, :, :].repeat(1, 3, 1, 1))

        return attentions, images

    def forward(self, input, gammas=None):
        feat, _ = self.compute_feats(input, gammas=gammas)
        attentions, images = self.compute_attention_content(feat)
        _, _, outputs = self.compute_outputs(input, attentions, images)

        o = outputs[0]
        for i in range(1, self.nb_mask_attn):
            o += outputs[i]
        return o
