import torch
import torch.nn as nn
from .blocks import FeatureFusionBlockMatrix, FeatureFusionBlockVector
import os


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
    pretrained.layer0 = nn.Sequential(
        model.conv_stem, model.bn1, model.act1, *model.blocks[0:2]
    )
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained


def _make_vit(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(
        model.patch_embed, model.pos_drop, *model.blocks[0:2]
    )
    pretrained.layer1 = nn.Sequential(*model.blocks[2:5])
    pretrained.layer2 = nn.Sequential(*model.blocks[5:8])
    pretrained.layer3 = nn.Sequential(*model.blocks[8:])
    return pretrained


def configure_forward_network(net, transpose=False):
    def forward(x):
        out0 = net.layer0(x)
        out1 = net.layer1(out0)
        out2 = net.layer2(out1)
        out3 = net.layer3(out2)
        if transpose:
            out0, out1, out2, out3 = (
                out0.transpose(2, 1).contiguous(),
                out1.transpose(2, 1).contiguous(),
                out2.transpose(2, 1).contiguous(),
                out3.transpose(2, 1).contiguous(),
            )
        return out0, out1, out2, out3

    net.forward = forward


def calc_channels(pretrained, inp_res=224):
    channels = []
    feats = []
    tmp = torch.zeros(1, 3, inp_res, inp_res)

    # forward pass
    outs = pretrained(tmp)
    for out in outs:
        channels.append(out.shape[1])
        feats.append(out.shape[2:])

    return channels, feats


def create_timm_model(model_name, config_path, weight_path):
    import timm

    model = timm.create_model(model_name, pretrained=True)
    return model


def create_segformer_model(model_name, config_path, weight_path):
    from mmseg.models import build_segmentor
    import mmcv

    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.train_cfg = None
    try:
        weights = torch.jit.load(weight_path).state_dict()
        print("Torch script weights are detected and loaded in %s" % weight_path)
    except:
        weights = torch.load(weight_path)

    segformer = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    model = segformer.backbone

    weights = {
        key.replace("backbone.", ""): value
        for (key, value) in weights.items()
        if "backbone." in key
    }

    model.load_state_dict(weights, strict=True)

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
        "make_function": None,  # unused
    },
    "vitbase": {
        "model_name": "vit_base_patch16_224",
        "create_model_function": create_timm_model,
        "make_function": _make_vit,
    },
    "vitsmall": {
        "model_name": "vit_small_patch16_224",
        "create_model_function": create_timm_model,
        "make_function": _make_vit,
    },
    "vitsmall2": {
        "model_name": "vit_small_r26_s32_224",
        "create_model_function": create_timm_model,
        "make_function": _make_vit,
    },
}


def _make_projector(projector_model, cout, proj_type, expand, config_path, weight_path):
    assert proj_type in [0, 1, 2], "Invalid projection type"

    ### Build pretrained feature network
    projector_gen = projector_models[projector_model]
    model = projector_gen["create_model_function"](
        projector_gen["model_name"], config_path, weight_path
    )
    if projector_model == "segformer":
        pretrained = model
    else:
        pretrained = projector_gen["make_function"](model)
        configure_forward_network(pretrained, transpose="vit" in projector_model)

    # determine resolution of feature maps, this is later used to calculate the number
    # of down blocks in the discriminators. Interestingly, the best results are achieved
    # by fixing this to 256, ie., we use the same number of down blocks per discriminator
    # independent of the dataset resolution

    pretrained.CHANNELS, pretrained.FEATS = calc_channels(pretrained)
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
        **kwargs
    ):
        super().__init__()
        self.proj_type = proj_type
        self.cout = cout
        self.expand = expand

        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(
            projector_model=projector_model,
            cout=self.cout,
            proj_type=self.proj_type,
            expand=self.expand,
            config_path=config_path,
            weight_path=weight_path,
        )

        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS
        self.FEATS = self.pretrained.FEATS

    def forward(self, x):
        # predict feature maps

        out0, out1, out2, out3 = self.pretrained(x)

        # start enumerating at the lowest layer (this is where we put the first discriminator)
        out = {
            "0": out0,
            "1": out1,
            "2": out2,
            "3": out3,
        }

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
