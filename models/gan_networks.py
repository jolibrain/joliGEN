import os
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
import math
import torchvision.models as models

from .modules.utils import (
    spectral_norm,
    init_net,
    init_weights,
    get_norm_layer,
    get_weights,
    download_segformer_weight,
)

from .modules.resnet_architecture.resnet_generator import ResnetGenerator
from .modules.unet_architecture.unet_generator import UnetGenerator
from .modules.resnet_architecture.resnet_generator import ResnetGenerator_attn
from .modules.discriminators import NLayerDiscriminator
from .modules.discriminators import PixelDiscriminator

from .modules.classifiers import (
    torch_model,
    TORCH_MODEL_CLASSES,
)

from .modules.cut_networks import PatchSampleF, PatchSampleF_QSAttn
from .modules.projected_d.discriminator import (
    ProjectedDiscriminator,
    TemporalProjectedDiscriminator,
)
from .modules.vision_aided_d import VisionAidedDiscriminator
from .modules.segformer.segformer_generator import (
    SegformerBackbone,
    SegformerGenerator_attn,
)
from .modules.ittr.ittr_generator import ITTRGenerator
from .modules.multimodal_encoder import E_ResNet, E_NLayers
from .modules.unet_generator_attn.unet_generator_attn import (
    UNet as UNet_mha,
    UViT as UViT,
)

from .modules.hdit.hdit import HDiT, HDiTConfig

from .modules.img2img_turbo.img2img_turbo import Img2ImgTurbo


def define_G(
    model_input_nc,
    model_output_nc,
    G_ngf,
    G_netG,
    G_nblocks,
    G_norm,
    G_dropout,
    G_spectral,
    model_init_type,
    model_init_gain,
    G_padding_type,
    data_crop_size,
    G_attn_nb_mask_attn,
    G_attn_nb_mask_input,
    jg_dir,
    G_config_segformer,
    G_backward_compatibility_twice_resnet_blocks,
    G_unet_mha_num_head_channels,
    G_unet_mha_res_blocks,
    G_unet_mha_channel_mults,
    G_unet_mha_norm_layer,
    G_unet_mha_group_norm_size,
    G_uvit_num_transformer_blocks,
    G_unet_mha_vit_efficient,
    G_hdit_depths,
    G_hdit_widths,
    G_hdit_patch_size,
    train_feat_wavelet,
    G_lora_unet,
    G_lora_vae,
    G_prompt,
    **unused_options,
):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        G_netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        G_dropout (bool) -- if use dropout layers.
        G_spectral (bool) -- if use spectral norm.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=G_norm)

    if G_netG == "resnet":
        net = ResnetGenerator(
            model_input_nc,
            model_output_nc,
            G_ngf,
            norm_layer=norm_layer,
            use_dropout=G_dropout,
            use_spectral=G_spectral,
            n_blocks=G_nblocks,
            padding_type=G_padding_type,
        )
    elif G_netG == "mobile_resnet":
        net = ResnetGenerator(
            model_input_nc,
            model_output_nc,
            ngf=G_ngf,
            norm_layer=norm_layer,
            n_blocks=G_nblocks,
            mobile=True,
        )
    elif G_netG == "unet_128":
        net = UnetGenerator(
            model_input_nc,
            model_output_nc,
            7,
            G_ngf,
            norm_layer=norm_layer,
            use_dropout=G_dropout,
        )
    elif G_netG == "unet_256":
        net = UnetGenerator(
            model_input_nc,
            model_output_nc,
            8,
            G_ngf,
            norm_layer=norm_layer,
            use_dropout=G_dropout,
        )
    elif G_netG == "resnet_attn":
        net = ResnetGenerator_attn(
            model_input_nc,
            model_output_nc,
            G_attn_nb_mask_attn,
            G_attn_nb_mask_input,
            G_ngf,
            n_blocks=G_nblocks,
            use_spectral=G_spectral,
            padding_type=G_padding_type,
            twice_resnet_blocks=G_backward_compatibility_twice_resnet_blocks,
            freq_space=train_feat_wavelet,
        )
    elif G_netG == "mobile_resnet_attn":
        net = ResnetGenerator_attn(
            model_input_nc,
            model_output_nc,
            G_attn_nb_mask_attn,
            G_attn_nb_mask_input,
            G_ngf,
            n_blocks=G_nblocks,
            use_spectral=G_spectral,
            padding_type=G_padding_type,
            mobile=True,
            twice_resnet_blocks=G_backward_compatibility_twice_resnet_blocks,
            freq_space=train_feat_wavelet,
        )
    elif G_netG == "segformer_attn_conv":
        net = SegformerGenerator_attn(
            jg_dir,
            G_config_segformer,
            model_input_nc,
            img_size=data_crop_size,
            nb_mask_attn=G_attn_nb_mask_attn,
            nb_mask_input=G_attn_nb_mask_input,
            final_conv=True,
            padding_type=G_padding_type,
        )
        return net
    elif G_netG == "segformer_conv":
        net = SegformerBackbone(
            jg_dir,
            G_config_segformer,
            model_input_nc,
            img_size=data_crop_size,
            num_classes=256,
            final_conv=True,
            padding_type=G_padding_type,
        )
        return net
    elif G_netG == "ittr":
        net = ITTRGenerator(
            model_input_nc,
            model_output_nc,
            img_size=data_crop_size,
            n_blocks=G_nblocks,
            ngf=G_ngf,
        )
        return net
    elif G_netG == "unet_mha":
        net = UNet_mha(
            image_size=data_crop_size,
            in_channel=model_input_nc,
            inner_channel=G_ngf,
            cond_embed_dim=G_ngf * 4,
            out_channel=model_output_nc,
            res_blocks=G_unet_mha_res_blocks,
            attn_res=[16],
            channel_mults=G_unet_mha_channel_mults,  # e.g. (1, 2, 4, 8)
            num_head_channels=G_unet_mha_num_head_channels,
            tanh=True,
            n_timestep_train=0,  # unused
            n_timestep_test=0,  # unused
            norm=G_unet_mha_norm_layer,
            group_norm_size=G_unet_mha_group_norm_size,
        )
        return net
    elif G_netG == "uvit":
        net = UViT(
            image_size=data_crop_size,
            in_channel=model_input_nc,
            inner_channel=G_ngf,
            cond_embed_dim=G_ngf * 4,
            out_channel=model_output_nc,
            res_blocks=G_unet_mha_res_blocks,
            attn_res=[16],
            channel_mults=G_unet_mha_channel_mults,  # e.g. (1, 2, 4, 8)
            num_head_channels=G_unet_mha_num_head_channels,
            tanh=True,
            n_timestep_train=0,  # unused
            n_timestep_test=0,  # unused
            norm=G_unet_mha_norm_layer,
            group_norm_size=G_unet_mha_group_norm_size,
            num_transformer_blocks=G_uvit_num_transformer_blocks,
            efficient=G_unet_mha_vit_efficient,
        )
        return net
    elif G_netG == "hdit":
        hdit_config = HDiTConfig(G_hdit_depths, G_hdit_widths, G_hdit_patch_size)
        print("HDiT levels=", hdit_config.levels)
        print("HDiT mapping=", hdit_config.mapping)
        net = HDiT(
            levels=hdit_config.levels,
            mapping=hdit_config.mapping,
            in_channels=model_input_nc,
            out_channels=model_output_nc,
            patch_size=hdit_config.patch_size,
            last_zero_init=False,
            num_classes=0,
            mapping_cond_dim=0,
        )
        cond_embed_dim = hdit_config.mapping.width
        net.cond_embed_dim = cond_embed_dim
        return net
    elif G_netG == "img2img_turbo":
        ##TODO: add img2img_turbo
        net = Img2ImgTurbo(
            ##TODO
            in_channels=model_input_nc,
            out_channels=model_output_nc,
            lora_rank_unet=G_lora_unet,
            lora_rank_vae=G_lora_vae,
            prompt=G_prompt,
        )
        return net
    else:
        raise NotImplementedError(
            "Generator model name [%s] is not recognized" % G_netG
        )
    return init_net(net, model_init_type, model_init_gain)


def define_D(
    D_netDs,
    model_input_nc,
    D_ndf,
    D_n_layers,
    D_norm,
    D_dropout,
    D_spectral,
    model_init_type,
    model_init_gain,
    D_no_antialias,
    data_crop_size,
    D_proj_network_type,
    D_proj_interp,
    D_proj_config_segformer,
    D_proj_weight_segformer,
    jg_dir,
    data_temporal_number_frames,
    data_temporal_frame_step,
    data_online_context_pixels,
    D_vision_aided_backbones,
    dataaug_D_diffusion,
    f_s_semantic_nclasses,
    model_depth_network,
    train_feat_wavelet,
    **unused_options,
):
    """Create a discriminator

    Parameters:
        model_input_nc (int)     -- the number of channels in input images
        D_ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        D_n_layers (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        D_norm (str)         -- the type of normalization layers used in the network.
        D_dropout (bool) -- whether to use dropout layers
        D_spectral(bool) -- whether to use spectral norm
        model_init_type (str)    -- the name of the initialization method.
        model_init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <D_n_layers> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=D_norm)

    margin = data_online_context_pixels * 2

    return_nets = {}

    img_size = data_crop_size

    for netD in D_netDs:
        if netD == "basic":  # default PatchGAN classifier
            net = NLayerDiscriminator(
                model_input_nc,
                D_ndf,
                n_layers=3,
                norm_layer=norm_layer,
                use_dropout=D_dropout,
                use_spectral=D_spectral,
                freq_space=train_feat_wavelet,
            )
            return_nets[netD] = init_net(net, model_init_type, model_init_gain)

        elif netD == "n_layers":  # more options
            net = NLayerDiscriminator(
                model_input_nc,
                D_ndf,
                D_n_layers,
                norm_layer=norm_layer,
                use_dropout=D_dropout,
                use_spectral=D_spectral,
            )
            return_nets[netD] = init_net(net, model_init_type, model_init_gain)

        elif netD == "pixel":  # classify if each pixel is real or fake
            net = PixelDiscriminator(model_input_nc, D_ndf, norm_layer=norm_layer)
            return_nets[netD] = init_net(net, model_init_type, model_init_gain)

        elif netD in TORCH_MODEL_CLASSES:  # load torchvision model
            nclasses = 1
            template = netD
            net = torch_model(
                model_input_nc,
                D_ndf,
                nclasses,
                data_crop_size + margin,
                template,
                pretrained=False,
            )
            return_nets[netD] = net

        elif netD == "projected_d":  # D in projected feature space
            if D_proj_network_type == "segformer":
                weight_path = os.path.join(jg_dir, D_proj_weight_segformer)
                if not os.path.exists(weight_path):
                    print(
                        "Downloading pretrained segformer weights for projected D feature extractor."
                    )
                    download_segformer_weight(weight_path)

            elif D_proj_network_type == "depth":
                weight_path = model_depth_network

            else:
                weight_path = ""
            net = ProjectedDiscriminator(
                D_proj_network_type,
                interp=D_proj_interp,
                config_path=os.path.join(jg_dir, D_proj_config_segformer),
                weight_path=weight_path,
                img_size=data_crop_size + margin,
                diffusion_aug=dataaug_D_diffusion,
            )
            return_nets[netD] = net  # no init since custom frozen backbone

        elif netD == "vision_aided":
            net = VisionAidedDiscriminator(cv_type=D_vision_aided_backbones)
            return_nets[netD] = net  # no init since partly frozen

        elif netD == "depth":  # default patch-based on depth
            net = NLayerDiscriminator(
                1,
                D_ndf,
                n_layers=3,
                norm_layer=norm_layer,
                use_dropout=D_dropout,
                use_spectral=D_spectral,
            )
            return_nets[netD] = init_net(net, model_init_type, model_init_gain)

        elif netD == "sam":  # default patch-based on sam
            net = NLayerDiscriminator(
                1,
                D_ndf,
                n_layers=3,
                norm_layer=norm_layer,
                use_dropout=D_dropout,
                use_spectral=D_spectral,
            )
            return_nets[netD] = init_net(net, model_init_type, model_init_gain)

        elif netD == "temporal":
            # projected D temporal
            weight_path = os.path.join(jg_dir, D_proj_weight_segformer)
            if D_proj_network_type == "segformer" and not os.path.exists(weight_path):
                print(
                    "Downloading pretrained segformer weights for projected D feature extractor."
                )
                download_segformer_weight(weight_path)
            net = TemporalProjectedDiscriminator(
                D_proj_network_type,
                interp=D_proj_interp,
                config_path=os.path.join(jg_dir, D_proj_config_segformer),
                weight_path=weight_path,
                data_temporal_number_frames=data_temporal_number_frames,
                data_temporal_frame_step=data_temporal_frame_step,
                img_size=data_crop_size + margin,
            )

            return_nets[netD] = net  # no init since custom frozen backbone

        elif netD == "mask":
            net = NLayerDiscriminator(
                f_s_semantic_nclasses,  # as number of input dimension, i.e. one-hot from gumbel-softmax
                D_ndf,
                n_layers=3,
                norm_layer=norm_layer,
                use_dropout=D_dropout,
                use_spectral=D_spectral,
            )
            return_nets[netD] = init_net(net, model_init_type, model_init_gain)

        else:
            raise NotImplementedError(
                "Discriminator model name [%s] is not recognized" % netD
            )

    return return_nets


def define_F(
    alg_cut_netF_nc,
    alg_cut_netF,
    alg_cut_netF_norm,
    alg_cut_netF_dropout,
    model_init_type,
    model_init_gain,
    **unused_options,
):
    if alg_cut_netF == "global_pool":
        net = PoolingF()
    elif alg_cut_netF == "sample":
        net = PatchSampleF(
            use_mlp=False,
            init_type=model_init_type,
            init_gain=model_init_gain,
            nc=alg_cut_netF_nc,
        )
    elif alg_cut_netF == "mlp_sample":
        net = PatchSampleF(
            use_mlp=True,
            init_type=model_init_type,
            init_gain=model_init_gain,
            nc=alg_cut_netF_nc,
        )
    elif alg_cut_netF == "sample_qsattn":
        net = PatchSampleF_QSAttn(
            use_mlp=False,
            init_type=model_init_type,
            init_gain=model_init_gain,
            nc=alg_cut_netF_nc,
        )
    elif alg_cut_netF == "mlp_sample_qsattn":
        net = PatchSampleF_QSAttn(
            use_mlp=True,
            init_type=model_init_type,
            init_gain=model_init_gain,
            nc=alg_cut_netF_nc,
        )
    else:
        raise NotImplementedError("projection model name [%s] is not recognized" % netF)
    return init_net(net, model_init_type, model_init_gain)


def define_E(
    model_input_nc,
    train_mm_nz,
    G_ngf,
    G_netE,
    model_init_type="xavier",
    model_init_gain=0.02,
    **unused_options,
):
    net = None
    vaeLike = False
    model_output_nc = train_mm_nz
    norm_layer = get_norm_layer(norm_type="batch")
    nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    if G_netE == "resnet_128":
        net = E_ResNet(
            model_input_nc,
            model_output_nc,
            G_ngf,
            n_blocks=4,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            vaeLike=vaeLike,
        )
    elif G_netE == "resnet_256":
        net = E_ResNet(
            model_input_nc,
            model_output_nc,
            G_ngf,
            n_blocks=5,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            vaeLike=vaeLike,
        )
    elif G_netE == "resnet_512":
        net = E_ResNet(
            model_input_nc,
            model_output_nc,
            G_ngf,
            n_blocks=6,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            vaeLike=vaeLike,
        )
    elif G_netE == "conv_128":
        net = E_NLayers(
            model_input_nc,
            model_output_nc,
            G_ngf,
            n_layers=4,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            vaeLike=vaeLike,
        )
    elif G_netE == "conv_256":
        net = E_NLayers(
            model_input_nc,
            model_output_nc,
            G_ngf,
            n_layers=5,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            vaeLike=vaeLike,
        )
    elif G_netE == "conv_512":
        net = E_NLayers(
            model_input_nc,
            model_output_nc,
            G_ngf,
            n_layers=5,
            norm_layer=norm_layer,
            nl_layer=nl_layer,
            vaeLike=vaeLike,
        )
    else:
        raise NotImplementedError("E encoder model name [%s] is not recognized" % net)

    return init_net(net, model_init_type, model_init_gain)
