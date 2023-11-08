import os

import random
import torch
import torch.nn as nn
from torchvision import transforms

from .blocks import FeatureFusionBlockMatrix, FeatureFusionBlockVector
from .diffusion import Diffusion
from models.modules.segformer.config import load_config_file
from models.modules.segformer.builder_from_scratch import JoliSegformer
from models.modules.utils import download_midas_weight


def _make_scratch_ccm(scratch, in_channels, cout, conv, expand=False):
    # shapes
    out_channels = [cout, cout * 2, cout * 4, cout * 8] if expand else [cout] * 4

    scratch.layer0_ccm = conv(
        in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True
    )
    scratch.layer1_ccm = conv(
        in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True
    )
    scratch.layer2_ccm = conv(
        in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True
    )
    scratch.layer3_ccm = conv(
        in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True
    )

    scratch.CHANNELS = out_channels

    return scratch


def _make_scratch_csm(scratch, in_channels, cout, fusion_block, expand):
    scratch.layer3_csm = fusion_block(
        in_channels[3], nn.ReLU(False), expand=expand, lowest=True
    )
    scratch.layer2_csm = fusion_block(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = fusion_block(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = fusion_block(in_channels[0], nn.ReLU(False))

    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout * 2, cout * 4] if expand else [cout] * 4

    return scratch


def _make_efficientnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, *model.blocks[0:2])
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    configure_forward_network(pretrained)
    pretrained.get_feats = pretrained.forward
    return pretrained


def _make_vit_timm(model):
    configure_get_feats_vit_timm(model)
    return model


def _make_vit_clip(model):
    configure_get_feats_vit_clip(model)
    return model


def _make_segformer(model):
    model.get_feats = model.forward
    return model


def _make_depth(model):
    configure_get_feats_depth(model)
    return model


def _make_dinov2(model):
    configure_get_feats_dinov2(model)
    return model


def configure_forward_network(net):
    def forward(x):
        out0 = net.layer0(x)
        out1 = net.layer1(out0)
        out2 = net.layer2(out1)
        out3 = net.layer3(out2)
        return out0, out1, out2, out3

    net.forward = forward


def configure_get_feats_vit_clip(net):
    num_layers = len(net.transformer.resblocks)

    def get_feats(x):
        x = net.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                net.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + net.positional_embedding.to(x.dtype)
        x = net.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        outs = []
        # num_layers = 12
        for i in range(num_layers):
            block = net.transformer.resblocks[i]
            x = block(x)
            if i in [2, 5, 8]:
                outs.append(x.permute(1, 0, 2).contiguous())
        outs.append(x.permute(1, 0, 2).contiguous())

        return outs

    net.get_feats = get_feats


def configure_get_feats_vit_timm(net):
    def get_feats(x):
        x = net.patch_embed(x)
        x = torch.cat((net.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = net.pos_drop(x + net.pos_embed)
        outs = []
        for i, block in enumerate(net.blocks):
            x = block(x)
            if i in [2, 5, 8]:
                outs.append(x.transpose(2, 1).contiguous())
        outs.append(x.transpose(2, 1).contiguous())

        return outs

    net.get_feats = get_feats


def configure_get_feats_depth(net):
    def get_feats(x):
        x = net.transform(x)

        if net.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = net.forward_transformer(net.pretrained, x)
        if net.number_layers == 3:
            layer_1, layer_2, layer_3 = layers
        else:
            layer_1, layer_2, layer_3, layer_4 = layers

        layer_1_rn = net.scratch.layer1_rn(layer_1)
        layer_2_rn = net.scratch.layer2_rn(layer_2)
        layer_3_rn = net.scratch.layer3_rn(layer_3)
        if net.number_layers >= 4:
            layer_4_rn = net.scratch.layer4_rn(layer_4)

        if net.number_layers == 3:
            path_3 = net.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        else:
            path_4 = net.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
            path_3 = net.scratch.refinenet3(
                path_4, layer_3_rn, size=layer_2_rn.shape[2:]
            )
        path_2 = net.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = net.scratch.refinenet1(path_2, layer_1_rn)

        if net.scratch.stem_transpose is not None:
            path_1 = net.scratch.stem_transpose(path_1)

        out = net.scratch.output_conv(path_1)

        outs = [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]

        return outs

    net.get_feats = get_feats


def configure_get_feats_dinov2(net):
    dino_layers = {
        "dinov2_vits14": [2, 5, 8, 11],
        "dinov2_vitb14": [3, 8, 12, 17],
        "dinov2_vitl14": [4, 10, 16, 23],
        "dinov2_vitg14": [6, 16, 26, 39],
        "dinov2_vits14_reg": [2, 5, 8, 11],
        "dinov2_vitb14_reg": [3, 8, 12, 17],
        "dinov2_vitl14_reg": [4, 10, 16, 23],
        "dinov2_vitg14_reg": [6, 16, 26, 39],
    }

    def get_feats(x):
        feats = net.get_intermediate_layers(
            x, n=[2, 5, 8, 11], return_class_token=False
        )
        return feats

    net.get_feats = get_feats


def calc_channels(pretrained, inp_res=224):
    channels = []
    feats = []
    tmp = torch.zeros(1, 3, inp_res, inp_res)

    # forward pass
    outs = pretrained.get_feats(tmp)
    for out in outs:
        channels.append(out.shape[1])
        feats.append(out.shape[2:])

    return channels, feats


def create_timm_model(model_name, config_path, weight_path, img_size):
    import timm

    if "vit" in model_name:
        model = timm.create_model(model_name, img_size=img_size, pretrained=True)
    else:
        model = timm.create_model(model_name, pretrained=True)
    return model


def create_clip_model(model_name, config_path, weight_path, img_size):
    import clip

    model = clip.load(model_name)

    return model[0].visual.float().cpu()


def create_dinov2_model(model_name, config_path, weight_path, img_size):
    dinov2_model = torch.hub.load(
        "facebookresearch/dinov2", model_name, force_reload=True
    )
    return dinov2_model


def create_segformer_model(model_name, config_path, weight_path, img_size):
    cfg = load_config_file(config_path)
    try:
        weights = torch.jit.load(weight_path).state_dict()
        print("Torch script weights are detected and loaded in %s" % weight_path)
    except:
        weights = torch.load(weight_path)

    if "state_dict" in weights:
        weights = weights["state_dict"]

    segformer = JoliSegformer(cfg)
    model = segformer.backbone

    weights = {
        key.replace("backbone.", ""): value
        for (key, value) in weights.items()
        if "backbone." in key
    }

    model.load_state_dict(weights, strict=True)

    return model


def create_depth_model(model_name, config_path, weight_path, img_size):
    model_type = weight_path
    model = download_midas_weight(model_type)

    input_size = 384
    if model_type == "MiDas_small" or model_type == "DPT_SwinV2_T_256":
        input_size = 256
    elif model_type == "DPT_BEiT_L_512":
        input_size = 512
    elif model_type == "DPT_LeViT_224":
        input_size = 224
    model.transform = transforms.Compose(
        [
            transforms.Resize(input_size),
        ]
    )

    return model


projector_models = {
    "efficientnet": {
        "model_name": "tf_efficientnet_lite0",
        "create_model_function": create_timm_model,
        "make_function": _make_efficientnet,
    },
    "segformer": {
        "model_name": "",  # unused
        "create_model_function": create_segformer_model,
        "make_function": _make_segformer,
    },
    "vitbase": {
        "model_name": "vit_base_patch16_224",
        "create_model_function": create_timm_model,
        "make_function": _make_vit_timm,
    },
    "vitsmall": {
        "model_name": "vit_small_patch16_224",
        "create_model_function": create_timm_model,
        "make_function": _make_vit_timm,
    },
    "vitsmall2": {
        "model_name": "vit_small_r26_s32_224",
        "create_model_function": create_timm_model,
        "make_function": _make_vit_timm,
    },
    "vitclip16": {
        "model_name": "ViT-B/16",
        "create_model_function": create_clip_model,
        "make_function": _make_vit_clip,
    },
    "vitclip14": {
        "model_name": "ViT-L/14@336px",
        "create_model_function": create_clip_model,
        "make_function": _make_vit_clip,
    },
    "depth": {
        "model_name": "",
        "create_model_function": create_depth_model,
        "make_function": _make_depth,
    },
    "dinov2_vits14": {
        "model_name": "dinov2_vits14",
        "create_model_function": create_dinov2_model,
        "make_function": _make_dinov2,
    },
    "dinov2_vitb14": {
        "model_name": "dinov2_vitb14",
        "create_model_function": create_dinov2_model,
        "make_function": _make_dinov2,
    },
    "dinov2_vitl14": {
        "model_name": "dinov2_vitl14",
        "create_model_function": create_dinov2_model,
        "make_function": _make_dinov2,
    },
    "dinov2_vitg14": {
        "model_name": "dinov2_vitg14",
        "create_model_function": create_dinov2_model,
        "make_function": _make_dinov2,
    },
    "dinov2_vits14_reg": {
        "model_name": "dinov2_vits14_reg",
        "create_model_function": create_dinov2_model,
        "make_function": _make_dinov2,
    },
    "dinov2_vitb14_reg": {
        "model_name": "dinov2_vitb14_reg",
        "create_model_function": create_dinov2_model,
        "make_function": _make_dinov2,
    },
    "dinov2_vitl14": {
        "model_name": "dinov2_vitl14_reg",
        "create_model_function": create_dinov2_model,
        "make_function": _make_dinov2,
    },
    "dinov2_vitg14_reg": {
        "model_name": "dinov2_vitg14_reg",
        "create_model_function": create_dinov2_model,
        "make_function": _make_dinov2,
    },
}


def _make_projector(
    projector_model, cout, proj_type, expand, config_path, weight_path, interp
):
    assert proj_type in [0, 1, 2], "Invalid projection type"

    ### Build pretrained feature network
    projector_gen = projector_models[projector_model]
    model = projector_gen["create_model_function"](
        projector_gen["model_name"], config_path, weight_path, interp
    )

    pretrained = projector_gen["make_function"](model)

    # determine resolution of feature maps, this is later used to calculate the number
    # of down blocks in the discriminators. Interestingly, the best results are achieved
    # by fixing this to 256, ie., we use the same number of down blocks per discriminator
    # independent of the dataset resolution

    pretrained.CHANNELS, pretrained.FEATS = calc_channels(pretrained, inp_res=interp)
    for feat in pretrained.FEATS:
        pretrained.RESOLUTIONS = [feat[0] for feat in pretrained.FEATS]

    if proj_type == 0:
        return pretrained, None

    ### Build CCM
    scratch = nn.Module()

    # When using we do not have features maps but features vectors, which leads to small differences.
    vit = "vit" in projector_model
    if vit:
        conv = nn.Conv1d
    else:
        conv = nn.Conv2d

    scratch = _make_scratch_ccm(
        scratch, in_channels=pretrained.CHANNELS, cout=cout, conv=conv, expand=expand
    )
    pretrained.CHANNELS = scratch.CHANNELS

    if proj_type == 1:
        return pretrained, scratch

    ### build CSM
    if vit:
        feature_block = FeatureFusionBlockVector
    else:
        feature_block = FeatureFusionBlockMatrix
    scratch = _make_scratch_csm(
        scratch,
        in_channels=scratch.CHANNELS,
        cout=cout,
        expand=expand,
        fusion_block=feature_block,
    )

    # CSM upsamples x2 so the feature map resolution doubles
    if not vit:  # interpolation is not possible for features vectors
        pretrained.RESOLUTIONS = [res * 2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch


class Proj(nn.Module):
    def __init__(
        self,
        projector_model,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        config_path="",
        weight_path="",
        interp=-1,
        img_size=256,
        diffusion_aug=False,
        **kwargs,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.cout = cout
        self.expand = expand

        if interp == -1:
            interp = img_size

        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(
            projector_model=projector_model,
            cout=self.cout,
            proj_type=self.proj_type,
            expand=self.expand,
            config_path=config_path,
            weight_path=weight_path,
            interp=interp,
        )

        if hasattr(self.pretrained, "model"):
            # To allow DDP
            self.model = self.pretrained.model

        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS
        self.FEATS = self.pretrained.FEATS

        self.diffusion_aug = diffusion_aug
        if self.diffusion_aug:
            self.diffusion = Diffusion(
                t_min=5, t_max=500, beta_start=1e-4, beta_end=1e-2
            )
            self.diffusion_noise_sd = 0.5

    def forward(self, x):
        # predict feature maps

        out0, out1, out2, out3 = self.pretrained.get_feats(x)

        # start enumerating at the lowest layer (this is where we put the first discriminator)
        out = {
            "0": out0,
            "1": out1,
            "2": out2,
            "3": out3,
        }

        # diffusion aug (first feature position)
        if self.diffusion_aug:
            out["0"] = self.diffusion(out["0"], noise_std=self.diffusion_noise_sd)
            out["1"] = self.diffusion(out["1"], noise_std=self.diffusion_noise_sd)
            out["2"] = self.diffusion(out["2"], noise_std=self.diffusion_noise_sd)
            out["3"] = self.diffusion(out["3"], noise_std=self.diffusion_noise_sd)

        if self.proj_type == 0:
            return out

        out0_channel_mixed = self.scratch.layer0_ccm(out["0"])
        out1_channel_mixed = self.scratch.layer1_ccm(out["1"])
        out2_channel_mixed = self.scratch.layer2_ccm(out["2"])
        out3_channel_mixed = self.scratch.layer3_ccm(out["3"])

        out = {
            "0": out0_channel_mixed,
            "1": out1_channel_mixed,
            "2": out2_channel_mixed,
            "3": out3_channel_mixed,
        }

        if self.proj_type == 1:
            return out

        # from bottom to top
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            "0": out0_scale_mixed,
            "1": out1_scale_mixed,
            "2": out2_scale_mixed,
            "3": out3_scale_mixed,
        }

        return out
