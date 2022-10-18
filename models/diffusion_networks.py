from .modules.utils import get_norm_layer


from .modules.unet_generator_attn.diffusion_generator import DiffusionGenerator


def define_G(
    model_input_nc,
    model_output_nc,
    G_netG,
    G_nblocks,
    data_crop_size,
    G_norm,
    G_unet_mha_n_timestep_train,
    G_unet_mha_n_timestep_test,
    G_ngf,
    G_unet_mha_num_head_channels,
    **unused_options
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

    if G_netG == "unet_mha":
        net = DiffusionGenerator(
            unet="unet_mha",
            image_size=data_crop_size,
            in_channel=model_input_nc * 2,
            inner_channel=G_ngf,  # e.g. 64 in palette repo
            out_channel=model_output_nc,
            res_blocks=G_nblocks,  # 2 in palette repo
            attn_res=[16],  # e.g.
            channel_mults=(1, 2, 4, 8),  # e.g.
            num_head_channels=G_unet_mha_num_head_channels,  # e.g. 32 in palette repo
            tanh=False,
            n_timestep_train=G_unet_mha_n_timestep_train,
            n_timestep_test=G_unet_mha_n_timestep_test,
        )
        return net
    else:
        raise NotImplementedError(
            "Generator model name [%s] is not recognized" % G_netG
        )
