import sys
import torch
import torch.nn as nn
from diffusers import AutoencoderDC
from .modules.utils import get_norm_layer
from .modules.diffusion_generator import DiffusionGenerator
from .modules.resnet_architecture.resnet_generator_diff import ResnetGenerator_attn_diff
from .modules.unet_generator_attn.unet_generator_attn import (
    UNet,
    UViT,
    UNetGeneratorRefAttn,
)
from .modules.unet_generator_attn.unet_generator_attn_vid import UNetVid

from .modules.hdit.hdit import HDiT, HDiTConfig

from .modules.palette_denoise_fn import PaletteDenoiseFn
from .modules.cm_generator import CMGenerator
from .modules.unet_generator_attn.unet_generator_attn_vid import UNetVid


class LatentWrapper(nn.Module):
    def __init__(
        self,
        model,
        dc_ae_path,
        dc_ae_torch_dtype,
        is_pix2pix,
        orig_model_output_nc,
        latent_dim,
        downsampling_factor,
    ):
        super().__init__()
        self.model = model
        self.is_pix2pix = is_pix2pix
        self.orig_model_output_nc = orig_model_output_nc
        self.latent_dim = latent_dim
        self.downsampling_factor = downsampling_factor

        self.dc_ae = AutoencoderDC.from_pretrained(
            dc_ae_path, torch_dtype=getattr(torch, dc_ae_torch_dtype)
        ).eval()
        self.dc_ae.requires_grad_(False)

    def parameters(self, recurse: bool = True):
        """Returns the parameters of the wrapped model, excluding the frozen autoencoder."""
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Returns the named parameters of the wrapped model, excluding the frozen autoencoder."""
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    def forward(self, *args, **kwargs):
        # Determine input images from args and kwargs
        y = kwargs.get("y_0", kwargs.get("y"))
        x = kwargs.get("y_cond", kwargs.get("x"))

        if y is None:
            if len(args) > 1:
                y = args[1]
            elif len(args) > 0:
                y = args[0]

        if x is None:
            if len(args) > 0:
                x = args[0]

        if y is None:
            raise ValueError(
                "LatentWrapper: could not determine the main image to process."
            )

        self.dc_ae.to(y.device)

        # Encode images to latent space
        mask = kwargs.get("mask")
        y_to_encode = y
        if mask is not None:
            y_to_encode = y * (1 - mask.float())

        y_latent = self.dc_ae.encode(y_to_encode.float()).latent

        if x is not None and x.dim() == 4 and x is not y:
            x_to_encode = x
            if mask is not None:
                x_to_encode = x * (1 - mask.float())
            x_latent = self.dc_ae.encode(x_to_encode.float()).latent
        else:
            x_latent = x

        # print('x_latent shape=', x_latent.shape)

        if mask is not None:
            # Downsize the mask using nearest neighbors
            downsampled_mask = torch.nn.functional.interpolate(
                mask.float(),
                size=(
                    mask.shape[2] // self.downsampling_factor,
                    mask.shape[3] // self.downsampling_factor,
                ),
                mode="nearest",
            )
            # Convert to single channel if necessary
            # if downsampled_mask.shape[1] > 1:
            #    downsampled_mask = downsampled_mask.mean(dim=1, keepdim=True)
            # Repeat across channels to match latent_dim
            kwargs["mask"] = downsampled_mask  # .repeat(1, self.latent_dim, 1, 1)
            # print('downsampled mask size=', kwargs['mask'].shape)

        # Reconstruct arguments for the wrapped model
        new_kwargs = kwargs.copy()
        if "y_0" in new_kwargs:
            new_kwargs["y_0"] = y_latent
        if "y" in new_kwargs:
            new_kwargs["y"] = y_latent
        if "y_cond" in new_kwargs:
            new_kwargs["y_cond"] = x_latent
        if "x" in new_kwargs:
            new_kwargs["x"] = x_latent

        new_args = list(args)
        if len(new_args) > 1:
            new_args[0] = y_latent
            new_args[1] = x_latent
        elif len(new_args) > 0:
            new_args[0] = y_latent  # Assuming single image is the main image

        # Call the wrapped model
        output = self.model(*new_args, **new_kwargs)

        if self.training:
            if isinstance(output, tuple) and len(output) == 3:
                noise, noise_hat, weight = output
                decoded_noise = self.dc_ae.decode(noise).sample
                decoded_noise_hat = self.dc_ae.decode(noise_hat).sample
                return decoded_noise, decoded_noise_hat, weight
        else:
            with torch.no_grad():
                output = self.dc_ae.decode(output).sample
                if self.orig_model_output_nc == 1:
                    output = output.mean(dim=1, keepdim=True)

        return output

    @property
    def beta_schedule(self):
        if hasattr(self.model, "denoise_fn"):  # DiffusionGenerator
            return self.model.denoise_fn.model.beta_schedule
        elif hasattr(self.model, "cm_model"):  # CMGenerator
            return self.model.cm_model.beta_schedule
        else:
            return self.model.beta_schedule

    @beta_schedule.setter
    def beta_schedule(self, value):
        if hasattr(self.model, "denoise_fn"):  # DiffusionGenerator
            self.model.denoise_fn.model.beta_schedule = value
        elif hasattr(self.model, "cm_model"):  # CMGenerator
            self.model.cm_model.beta_schedule = value
        else:
            self.model.beta_schedule = value

    def restoration(
        self,
        y_cond,
        y_t=None,
        y_0=None,
        mask=None,
        sample_num=8,
        cls=None,
        ref=None,
        guidance_scale=0.0,
        ddim_num_steps=10,
        ddim_eta=0.5,
    ):
        self.dc_ae.to(y_cond.device)

        # Encode image-like inputs
        if y_cond is not None and y_cond.dim() == 4:
            y_cond_to_encode = y_cond
            if mask is not None:
                y_cond_to_encode = y_cond * (1 - mask.float())
            y_cond_latent = self.dc_ae.encode(y_cond_to_encode.float()).latent
        else:
            y_cond_latent = y_cond

        if y_t is not None and y_t.dim() == 4:
            y_t_to_encode = y_t
            if mask is not None:
                y_t_to_encode = y_t * (1 - mask.float())
            y_t_latent = self.dc_ae.encode(y_t_to_encode.float()).latent
        else:
            y_t_latent = y_t

        if y_0 is not None and y_0.dim() == 4:
            y_0_to_encode = y_0
            if mask is not None:
                y_0_to_encode = y_0 * (1 - mask.float())
            y_0_latent = self.dc_ae.encode(y_0_to_encode.float()).latent
        else:
            y_0_latent = y_0

        if mask is not None and mask.dim() == 4:
            # Downsize the mask using nearest neighbors
            downsampled_mask = torch.nn.functional.interpolate(
                mask.float(),
                size=(
                    mask.shape[2] // self.downsampling_factor,
                    mask.shape[3] // self.downsampling_factor,
                ),
                mode="nearest",
            )
            # Convert to single channel if necessary (e.g., take mean across channels)
            # if downsampled_mask.shape[1] > 1:
            #    downsampled_mask = downsampled_mask.mean(dim=1, keepdim=True) # Take mean across channels
            # Repeat across channels to match latent_dim
            mask_latent = downsampled_mask.repeat(1, self.latent_dim, 1, 1)
            # print('mask latent size=', mask_latent.size())
        else:
            mask_latent = mask

        # Call the wrapped model's restoration method
        latent_output, latent_visuals = self.model.restoration(
            y_cond=y_cond_latent,
            y_t=y_t_latent,
            y_0=y_0_latent,
            mask=mask_latent,
            sample_num=sample_num,
            cls=cls,
            ref=ref,
            guidance_scale=guidance_scale,
            ddim_num_steps=ddim_num_steps,
            ddim_eta=ddim_eta,
        )

        # Decode the outputs
        with torch.no_grad():
            output = self.dc_ae.decode(latent_output).sample
            visuals = self.dc_ae.decode(latent_visuals).sample

            if self.orig_model_output_nc == 1:
                output = output.mean(dim=1, keepdim=True)
                visuals = visuals.mean(dim=1, keepdim=True)

            if mask is not None and y_cond is not None:
                output = output * mask.float() + y_cond * (1 - mask.float())

        return output, visuals


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
    G_unet_vid_max_sequence_length,
    G_unet_vid_cross_attention_dim,
    G_unet_vid_num_attention_heads,
    G_unet_vid_num_transformer_blocks,
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
    alg_diffusion_latent_dc_ae_path="",
    alg_diffusion_latent_dc_ae_torch_dtype="float32",
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

    is_latent_wrapper_enabled = (
        alg_diffusion_latent_dc_ae_path is not None
        and alg_diffusion_latent_dc_ae_path != ""
    )

    latent_data_crop_size = data_crop_size
    if is_latent_wrapper_enabled:
        # For dc-ae-f64c128-in-1.0-diffusers, latent dim is 32 and downsampling factor is 8
        latent_dim = 32
        downsampling_factor = 32

        is_pix2pix = alg_diffusion_task == "pix2pix"
        if model_type != "palette":
            is_pix2pix = is_pix2pix or (
                alg_diffusion_cond_embed != "" and alg_diffusion_cond_embed != "y_t"
            )

        if is_pix2pix:
            in_channel = latent_dim * 2
        else:
            in_channel = (
                latent_dim * 2
            )  # conditioning (akin to input_nc + output_nc in latent space)

        latent_model_output_nc = latent_dim
        latent_data_crop_size = data_crop_size // downsampling_factor
    else:
        if model_type == "palette":
            in_channel = model_input_nc + model_output_nc
        else:  # CM
            in_channel = model_input_nc
            if (
                alg_diffusion_cond_embed != "" and alg_diffusion_cond_embed != "y_t"
            ) or alg_diffusion_task == "pix2pix":
                in_channel = model_input_nc + model_output_nc
        latent_model_output_nc = model_output_nc

    if "mask" in alg_diffusion_cond_embed:
        in_channel += alg_diffusion_cond_embed_dim

    if G_netG == "unet_mha":
        if model_prior_321_backwardcompatibility:
            cond_embed_dim = G_ngf * 4
        else:
            cond_embed_dim = alg_diffusion_cond_embed_dim

        model = UNet(
            image_size=latent_data_crop_size,
            in_channel=in_channel,
            inner_channel=G_ngf,
            out_channel=latent_model_output_nc,
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

    elif G_netG == "unet_vid":
        cond_embed_dim = alg_diffusion_cond_embed_dim

        model = UNetVid(
            image_size=latent_data_crop_size,
            in_channel=in_channel,
            inner_channel=G_ngf,
            out_channel=latent_model_output_nc,
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
            max_sequence_length=G_unet_vid_max_sequence_length,
            cross_attention_dim=G_unet_vid_cross_attention_dim,
            num_attention_heads=G_unet_vid_num_attention_heads,
            num_transformer_blocks=G_unet_vid_num_transformer_blocks,
        )

    elif G_netG == "unet_mha_ref_attn":
        cond_embed_dim = alg_diffusion_cond_embed_dim

        model = UNetGeneratorRefAttn(
            image_size=latent_data_crop_size,
            in_channel=in_channel,
            inner_channel=G_ngf,
            out_channel=latent_model_output_nc,
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
            image_size=latent_data_crop_size,
            in_channel=in_channel,
            inner_channel=G_ngf,
            out_channel=latent_model_output_nc,
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
            output_nc=latent_model_output_nc,
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
        model = HDiT(
            levels=hdit_config.levels,
            mapping=hdit_config.mapping,
            in_channel=in_channel,
            out_channel=latent_model_output_nc,
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
            image_size=latent_data_crop_size,
            G_ngf=G_ngf,
            loading_backward_compatibility=model_prior_321_backwardcompatibility,
        )
    elif model_type == "cm" or model_type == "cm_gan":
        net = CMGenerator(
            cm_model=model,
            sampling_method="",
            image_size=latent_data_crop_size,
            G_ngf=G_ngf,
        )
    else:
        raise NotImplementedError(model_type + " not implemented")

    if is_latent_wrapper_enabled:
        is_pix2pix = alg_diffusion_task == "pix2pix"
        if model_type != "palette":
            is_pix2pix = is_pix2pix or (
                alg_diffusion_cond_embed != "" and alg_diffusion_cond_embed != "y_t"
            )

        net = LatentWrapper(
            net,
            alg_diffusion_latent_dc_ae_path,
            alg_diffusion_latent_dc_ae_torch_dtype,
            is_pix2pix,
            model_output_nc,
            latent_dim,
            downsampling_factor,
        )

    return net
