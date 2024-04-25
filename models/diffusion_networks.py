import sys
from .modules.utils import get_norm_layer
from .modules.diffusion_generator import DiffusionGenerator
from .modules.resnet_architecture.resnet_generator_diff import ResnetGenerator_attn_diff
from .modules.unet_generator_attn.unet_generator_attn import (
    UNet,
    UViT,
    UNetGeneratorRefAttn,
)
from .modules.hdit.hdit import HDiT, HDiTConfig

from .modules.palette_denoise_fn import PaletteDenoiseFn
from .modules.cm_generator import CMGenerator


def define_G(
    model_type,
    model_input_nc,
    model_output_nc,
    G_netG,
    G_nblocks,
    data_crop_size,
    G_norm,
    G_diff_n_timestep_train,
    G_diff_n_timestep_test,
    G_dropout,
    G_ngf,
    G_unet_mha_num_heads,
    G_unet_mha_num_head_channels,
    G_unet_mha_res_blocks,
    G_unet_mha_channel_mults,
    G_unet_mha_attn_res,
    G_hdit_depths,
    G_hdit_widths,
    G_hdit_patch_size,
    G_attn_nb_mask_attn,
    G_attn_nb_mask_input,
    G_spectral,
    jg_dir,
    G_padding_type,
    G_config_segformer,
    G_unet_mha_norm_layer,
    G_unet_mha_group_norm_size,
    G_uvit_num_transformer_blocks,
    G_unet_mha_vit_efficient,
    alg_palette_sampling_method,
    alg_diffusion_task,
    alg_diffusion_cond_embed,
    alg_diffusion_cond_embed_dim,
    alg_diffusion_ref_embed_net,
    model_prior_321_backwardcompatibility,
    dropout=0,
    channel_mults=(1, 2, 4, 8),
    conv_resample=True,
    use_checkpoint=False,
    use_fp16=False,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=True,
    use_new_attention_order=False,
    f_s_semantic_nclasses=-1,
    train_feat_wavelet=False,
    **unused_options,
):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        G_netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        G_norm (str) -- the name of normalization layers used in the network: batch | instance | none

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

    if model_type == "palette":
        in_channel = model_input_nc * 2
    else:  # CM
        in_channel = model_input_nc
        if (
            alg_diffusion_cond_embed != "" and alg_diffusion_cond_embed != "y_t"
        ) or alg_diffusion_task == "pix2pix":
            in_channel *= 2

    if "mask" in alg_diffusion_cond_embed:
        in_channel += alg_diffusion_cond_embed_dim

    if G_netG == "unet_mha":
        if model_prior_321_backwardcompatibility:
            cond_embed_dim = G_ngf * 4
        else:
            cond_embed_dim = alg_diffusion_cond_embed_dim

        model = UNet(
            image_size=data_crop_size,
            in_channel=in_channel,
            inner_channel=G_ngf,
            out_channel=model_output_nc,
            res_blocks=G_unet_mha_res_blocks,
            attn_res=G_unet_mha_attn_res,
            num_heads=G_unet_mha_num_heads,
            num_head_channels=G_unet_mha_num_head_channels,
            tanh=False,
            dropout=G_dropout,
            n_timestep_train=G_diff_n_timestep_train,
            n_timestep_test=G_diff_n_timestep_test,
            channel_mults=G_unet_mha_channel_mults,
            norm=G_unet_mha_norm_layer,
            group_norm_size=G_unet_mha_group_norm_size,
            efficient=G_unet_mha_vit_efficient,
            cond_embed_dim=cond_embed_dim,
            freq_space=train_feat_wavelet,
        )

    elif G_netG == "unet_mha_ref_attn":
        cond_embed_dim = alg_diffusion_cond_embed_dim

        model = UNetGeneratorRefAttn(
            image_size=data_crop_size,
            in_channel=in_channel,
            inner_channel=G_ngf,
            out_channel=model_output_nc,
            res_blocks=G_unet_mha_res_blocks,
            attn_res=G_unet_mha_attn_res,
            num_heads=G_unet_mha_num_heads,
            num_head_channels=G_unet_mha_num_head_channels,
            tanh=False,
            dropout=G_dropout,
            n_timestep_train=G_diff_n_timestep_train,
            n_timestep_test=G_diff_n_timestep_test,
            channel_mults=G_unet_mha_channel_mults,
            norm=G_unet_mha_norm_layer,
            group_norm_size=G_unet_mha_group_norm_size,
            efficient=G_unet_mha_vit_efficient,
            cond_embed_dim=cond_embed_dim,
            freq_space=train_feat_wavelet,
        )

    elif G_netG == "uvit":
        model = UViT(
            image_size=data_crop_size,
            in_channel=in_channel,
            inner_channel=G_ngf,
            out_channel=model_output_nc,
            res_blocks=G_unet_mha_res_blocks,
            attn_res=G_unet_mha_attn_res,
            num_heads=G_unet_mha_num_heads,
            num_head_channels=G_unet_mha_num_head_channels,
            tanh=False,
            dropout=G_dropout,
            n_timestep_train=G_diff_n_timestep_train,
            n_timestep_test=G_diff_n_timestep_test,
            channel_mults=G_unet_mha_channel_mults,
            norm=G_unet_mha_norm_layer,
            group_norm_size=G_unet_mha_group_norm_size,
            num_transformer_blocks=G_uvit_num_transformer_blocks,
            efficient=G_unet_mha_vit_efficient,
            cond_embed_dim=alg_diffusion_cond_embed_dim,
            freq_space=train_feat_wavelet,
        )
        cond_embed_dim = alg_diffusion_cond_embed_dim

    elif G_netG == "resnet_attn" or G_netG == "mobile_resnet_attn":
        mobile = "mobile" in G_netG
        G_ngf = alg_diffusion_cond_embed_dim
        model = ResnetGenerator_attn_diff(
            input_nc=in_channel,
            output_nc=model_output_nc,
            nb_mask_attn=G_attn_nb_mask_attn,
            nb_mask_input=G_attn_nb_mask_input,
            n_timestep_train=G_diff_n_timestep_train,
            n_timestep_test=G_diff_n_timestep_test,
            ngf=G_ngf,
            n_blocks=G_nblocks,
            use_spectral=False,
            padding_type="reflect",
            mobile=mobile,
            use_scale_shift_norm=True,
        )
        cond_embed_dim = alg_diffusion_cond_embed_dim
    elif G_netG == "hdit":
        hdit_config = HDiTConfig(G_hdit_depths, G_hdit_widths, G_hdit_patch_size)
        print("HDiT levels=", hdit_config.levels)
        print("HDiT mapping=", hdit_config.mapping)
        model = HDiT(
            levels=hdit_config.levels,
            mapping=hdit_config.mapping,
            in_channels=in_channel,
            out_channels=model_output_nc,
            patch_size=hdit_config.patch_size,
            num_classes=0,
            mapping_cond_dim=0,
            n_timestep_train=G_diff_n_timestep_train,
            n_timestep_test=G_diff_n_timestep_test,
        )
        cond_embed_dim = hdit_config.mapping.width
        model.cond_embed_dim = cond_embed_dim

    else:
        raise NotImplementedError(
            "Generator model name [%s] is not recognized" % G_netG
        )

    if model_type == "palette":
        denoise_fn = PaletteDenoiseFn(
            model=model,
            cond_embed_dim=cond_embed_dim,
            ref_embed_net=alg_diffusion_ref_embed_net,
            conditioning=alg_diffusion_cond_embed,
            nclasses=f_s_semantic_nclasses,
        )

        net = DiffusionGenerator(
            denoise_fn=denoise_fn,
            sampling_method=alg_palette_sampling_method,
            image_size=data_crop_size,
            G_ngf=G_ngf,
            loading_backward_compatibility=model_prior_321_backwardcompatibility,
        )
    elif model_type == "cm":
        net = CMGenerator(
            cm_model=model,
            sampling_method="",
            image_size=data_crop_size,
            G_ngf=G_ngf,
        )
    else:
        raise NotImplementedError(model_type + " not implemented")

    return net
