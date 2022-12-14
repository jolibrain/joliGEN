from .modules.utils import get_norm_layer


from .modules.diffusion_generator import DiffusionGenerator
from .modules.resnet_architecture.resnet_generator_diff import ResnetGenerator_attn_diff
from .modules.unet_generator_attn.unet_generator_attn import UNet
from .modules.segformer.segformer_diff import SegformerGeneratorDiff_attn


def define_G(
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
    G_unet_mha_num_head_channels,
    G_attn_nb_mask_attn,
    G_attn_nb_mask_input,
    G_spectral,
    jg_dir,
    G_padding_type,
    G_config_segformer,
    dropout=0,
    channel_mults=(1, 2, 4, 8),
    conv_resample=True,
    use_checkpoint=False,
    use_fp16=False,
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    resblock_updown=True,
    use_new_attention_order=False,
    **unused_options
):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        G_netG (str) -- the architecture's name: resnet_9blocks | resnet6blocks | unet_256 | unet_128
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

    if G_netG == "unet_mha":
        denoise_fn = UNet(
            image_size=data_crop_size,
            in_channel=model_input_nc * 2,
            inner_channel=G_ngf,
            out_channel=model_output_nc,
            res_blocks=G_nblocks,
            attn_res=[16],  # e.g.
            tanh=False,
            n_timestep_train=G_diff_n_timestep_train,
            n_timestep_test=G_diff_n_timestep_test,
            channel_mults=(1, 2, 4, 8),  # e.g.
            num_head_channels=G_unet_mha_num_head_channels,  # e.g. 32 in palette repo
        )

    elif G_netG == "resnet_attn" or G_netG == "mobile_resnet_attn":
        mobile = "mobile" in G_netG
        denoise_fn = ResnetGenerator_attn_diff(
            input_nc=model_input_nc * 2,
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

    elif G_netG == "segformer_attn_conv":
        denoise_fn = SegformerGeneratorDiff_attn(
            jg_dir,
            G_config_segformer,
            model_input_nc * 2 + 1,
            img_size=data_crop_size,
            nb_mask_attn=G_attn_nb_mask_attn,
            nb_mask_input=G_attn_nb_mask_input,
            inner_channel=G_ngf,
            n_timestep_train=G_diff_n_timestep_train,
            n_timestep_test=G_diff_n_timestep_test,
            final_conv=True,
            padding_type=G_padding_type,
        )

    else:

        raise NotImplementedError(
            "Generator model name [%s] is not recognized" % G_netG
        )

    net = DiffusionGenerator(
        denoise_fn=denoise_fn,
    )

    return net
