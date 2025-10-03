import math
import sys
import torch
import torch.nn as nn
from diffusers import AutoencoderDC, AutoencoderKL
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
from torchvision.utils import save_image
from .modules.diffusion_utils import predict_start_from_noise


class AutoencoderWrapper(AutoencoderDC):

    def set_scaling_factor(self, scaling_factor):
        self.scaling_factor = scaling_factor
    
    def encode(self, x):
        x = super().encode(x)
        x.latent /= self.scaling_factor
        return x

    def decode(self, x):
        x *= self.scaling_factor
        decoded_x = super().decode(x)
        # x = torch.clamp(127.5 * decoded_x + 128.0, 0, 255).to(dtype=torch.uint8)
        # return x
        return decoded_x


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
        finetune_decoder=False,
        dc_ae_scaling=1.0,
    ):
        super().__init__()
        self.model = model
        self.is_pix2pix = is_pix2pix
        self.orig_model_output_nc = orig_model_output_nc
        self.latent_dim = latent_dim
        self.downsampling_factor = downsampling_factor

        self.finetune_decoder = finetune_decoder
        self.dc_ae = AutoencoderWrapper.from_pretrained(
            dc_ae_path,
            torch_dtype=getattr(torch, dc_ae_torch_dtype),
        ).eval()
        self.dc_ae.set_scaling_factor(dc_ae_scaling)

        self.dc_ae.requires_grad_(False)
        if self.finetune_decoder:
            self.model.requires_grad_(False)
            self.dc_ae.decoder.requires_grad_(True)
        else:
            self.model.requires_grad_(True)

        self.alg_diffusion_latent_mask = False
        self._current_t = getattr(self.model, "current_t", None)

    def parameters(self, recurse: bool = True):
        """Returns the parameters of the wrapped model, excluding the frozen autoencoder."""
        if hasattr(self, "finetune_decoder") and self.finetune_decoder:
            return self.dc_ae.decoder.parameters(recurse=recurse)
        return self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Returns the named parameters of the wrapped model, excluding the frozen autoencoder."""
        if hasattr(self, "finetune_decoder") and self.finetune_decoder:
            return self.dc_ae.decoder.named_parameters(prefix=prefix, recurse=recurse)
        return self.model.named_parameters(prefix=prefix, recurse=recurse)

    @property
    def current_t(self):
        if hasattr(self.model, "current_t"):
            return self.model.current_t
        return self._current_t

    @current_t.setter
    def current_t(self, value):
        self._current_t = value
        if hasattr(self.model, "current_t"):
            self.model.current_t = value

    @property
    def sampling_method(self):
        return self.model.sampling_method

    @sampling_method.setter
    def sampling_method(self, value):
        self.model.sampling_method = value

    @property
    def denoise_fn(self):
        return self.model.denoise_fn

    @denoise_fn.setter
    def denoise_fn(self, value):
        self.model.denoise_fn = value

    def set_new_sampling_method(self, sampling_method):
        self.model.set_new_sampling_method(sampling_method)

    @property
    def ddim_num_steps(self):
        return self.model.ddim_num_steps

    @ddim_num_steps.setter
    def ddim_num_steps(self, value):
        self.model.ddim_num_steps = value

    def _resize_mask_to_latent(self, mask, latent_tensor):
        if mask is None:
            return None

        mask_latent = torch.nn.functional.interpolate(
            mask.float(),
            size=(latent_tensor.shape[2], latent_tensor.shape[3]),
            mode="nearest",
        )

        latent_channels = latent_tensor.shape[1]
        mask_channels = mask_latent.shape[1]

        if mask_channels == 1 and latent_channels > 1:
            mask_latent = mask_latent.expand(-1, latent_channels, -1, -1)
        elif mask_channels != latent_channels:
            repeat_factor = math.ceil(latent_channels / mask_channels)
            mask_latent = mask_latent.repeat(1, repeat_factor, 1, 1)[
                :, :latent_channels, :, :
            ]

        return mask_latent

    def compute_palette_loss(self, palette_model):
        y_0 = palette_model.gt_image
        y_cond = palette_model.cond_image
        mask = palette_model.mask
        noise = None
        cls = palette_model.cls
        self.alg_diffusion_latent_mask = palette_model.opt.alg_diffusion_latent_mask
        
        if self.finetune_decoder:
            palette_model.opt.alg_diffusion_lambda_G_pixel = 1.0

        if palette_model.opt.alg_diffusion_dropout_prob > 0.0:
            drop_ids = (
                torch.rand(mask.shape[0], device=mask.device)
                < palette_model.opt.alg_diffusion_dropout_prob
            )
        else:
            drop_ids = None

        if drop_ids is not None:
            if mask is not None:
                mask = torch.where(
                    drop_ids.reshape(-1, 1, 1, 1).expand(mask.shape),
                    palette_model.num_classes - 1,
                    mask,
                )
            if cls is not None:
                cls = torch.where(drop_ids, palette_model.num_classes - 1, cls)

        if palette_model.use_ref:
            ref = palette_model.ref_A
        else:
            ref = None

        self.dc_ae.to(y_0.device)
        # if mask is not None:
        # noise_for_mask = torch.randn_like(y_0)
        # y_0 = y_0 * (1 - mask.float()) + noise_for_mask * mask.float()
        y_latent = self.dc_ae.encode(y_0.float()).latent
        if mask is not None:
            #noise_for_mask = torch.randn_like(y_cond)
            y_cond = y_cond * (1 - mask.float()) + 0.5 * mask.float()
        x_latent = self.dc_ae.encode(y_cond.float()).latent

        downsampled_mask = None
        if mask is not None and self.alg_diffusion_latent_mask:
            downsampled_mask = self._resize_mask_to_latent(mask, y_latent)
        
        noise, noise_hat, min_snr_loss_weight = self.model(
            y_0=y_latent,
            y_cond=x_latent,
            noise=noise,
            mask=downsampled_mask,
            cls=cls,
            ref=ref,
        )

        if not palette_model.opt.alg_palette_minsnr:
            min_snr_loss_weight = 1.0

        if downsampled_mask is not None:
            mask_binary = torch.clamp(downsampled_mask, min=0, max=1)
            loss = palette_model.loss_fn(
                min_snr_loss_weight * mask_binary * noise,
                min_snr_loss_weight * mask_binary * noise_hat,
            )
        else:

            # debug for restoration
            self.loss_fn = palette_model.loss_fn

            loss = palette_model.loss_fn(
                min_snr_loss_weight * noise, min_snr_loss_weight * noise_hat
            )

        if isinstance(loss, dict):
            loss_tot = torch.zeros(size=(), device=noise.device)
            for cur_size, cur_loss in loss.items():
                setattr(palette_model, "loss_G_" + cur_size, cur_loss)
                loss_tot += cur_loss
            loss = loss_tot

        palette_model.loss_G_tot = palette_model.opt.alg_diffusion_lambda_G * loss

        if (
            hasattr(palette_model.opt, "alg_diffusion_lambda_G_pixel")
            and palette_model.opt.alg_diffusion_lambda_G_pixel > 0
        ):
            y_0_hat_latent = predict_start_from_noise(
                self.model.denoise_fn.model,
                self.model.y_t,
                self.model.t,
                noise_hat,
                phase="train",
            )
            y_0_hat_image = self.dc_ae.decode(y_0_hat_latent).sample
            loss_pixel = palette_model.loss_fn(palette_model.gt_image, y_0_hat_image)
            setattr(palette_model, "loss_G_pixel", loss_pixel)
            palette_model.loss_G_tot += (
                palette_model.opt.alg_diffusion_lambda_G_pixel * loss_pixel
            )

    def compute_cm_loss(self, cm_model):
        
        def pseudo_huber_loss(input_tensor, target_tensor):
            c = 0.00054 * math.sqrt(math.prod(input_tensor.shape[1:]))
            return torch.sqrt((input_tensor - target_tensor) ** 2 + c**2) - c

        y_0 = cm_model.gt_image
        y_cond = cm_model.cond_image
        mask = cm_model.mask

        self.dc_ae.to(y_0.device)
        self.alg_diffusion_latent_mask = getattr(
            cm_model.opt, "alg_diffusion_latent_mask", False
        )

        y_latent = self.dc_ae.encode(y_0.float()).latent
        if y_cond is not None:
            x_cond_latent = self.dc_ae.encode(y_cond.float()).latent
        else:
            x_cond_latent = None

        mask_latent = self._resize_mask_to_latent(mask, y_latent) if mask is not None else None

        mask_for_model = mask_latent if self.alg_diffusion_latent_mask else None

        (
            pred_latent,
            target_latent,
            num_timesteps,
            sigmas,
            loss_weights,
            next_noisy_latent,
            current_noisy_latent,
        ) = self.model(
            y_latent,
            cm_model.total_t,
            mask_for_model,
            x_cond_latent,
        )

        if mask_latent is not None:
            mask_latent = torch.clamp(mask_latent, min=0.0, max=1.0)
            pred_for_loss = mask_latent * pred_latent
            target_for_loss = mask_latent * target_latent
        else:
            pred_for_loss = pred_latent
            target_for_loss = target_latent

        loss = (
            pseudo_huber_loss(pred_for_loss, target_for_loss) * loss_weights
        ).mean()

        cm_model.loss_G_tot = cm_model.opt.alg_diffusion_lambda_G * loss

        pred_x = self.dc_ae.decode(pred_latent.clone()).sample

        with torch.no_grad():
            cm_model.next_noisy_x = self.dc_ae.decode(next_noisy_latent.clone()).sample
            cm_model.current_noisy_x = self.dc_ae.decode(current_noisy_latent.clone()).sample

        cm_model.pred_x = pred_x

        if mask is not None:
            mask_pred_x = mask * pred_x
        else:
            mask_pred_x = pred_x

        cm_model.loss_G_perceptual_lpips = 0
        cm_model.loss_G_perceptual_dists = 0
        cm_model.loss_G_perceptual = 0

        if "LPIPS" in cm_model.opt.alg_cm_perceptual_loss:
            if pred_x.size(1) > 3:
                cm_model.loss_G_perceptual_lpips = 0.0
                for channel in range(mask_pred_x.size(1)):
                    y_0_channel = y_0[:, channel, :, :].unsqueeze(1)
                    pred_channel = mask_pred_x[:, channel, :, :].unsqueeze(1)
                    cm_model.loss_G_perceptual_lpips += cm_model.criterionLPIPS(
                        y_0_channel, pred_channel
                    )
            else:
                cm_model.loss_G_perceptual_lpips = torch.mean(
                    cm_model.criterionLPIPS(y_0, mask_pred_x)
                )

        if "DISTS" in cm_model.opt.alg_cm_perceptual_loss:
            if pred_x.size(1) > 3:
                cm_model.loss_G_perceptual_dists = 0.0
                for channel in range(mask_pred_x.size(1)):
                    y_0_channel = y_0[:, channel, :, :].unsqueeze(1)
                    pred_channel = mask_pred_x[:, channel, :, :].unsqueeze(1)
                    cm_model.loss_G_perceptual_dists += cm_model.criterionDISTS(
                        y_0_channel, pred_channel
                    )
            else:
                cm_model.loss_G_perceptual_dists = cm_model.criterionDISTS(
                    y_0, mask_pred_x
                )

        if (
            cm_model.loss_G_perceptual_lpips > 0
            or cm_model.loss_G_perceptual_dists > 0
        ):
            cm_model.loss_G_perceptual = (
                cm_model.opt.alg_cm_lambda_perceptual
                * (cm_model.loss_G_perceptual_lpips + cm_model.loss_G_perceptual_dists)
            )
            cm_model.loss_G_cm = cm_model.loss_G_tot.clone()
            cm_model.loss_G_tot += cm_model.loss_G_perceptual

    def restoration_cm(
        self,
        y,
        y_cond,
        sigmas,
        mask=None,
        clip_denoised=True,
        latent_mask=False,
    ):
        self.dc_ae.to(y.device)
        self.alg_diffusion_latent_mask = latent_mask

        y_latent = self.dc_ae.encode(y.float()).latent
        if y_cond is not None:
            x_cond_latent = self.dc_ae.encode(y_cond.float()).latent
        else:
            x_cond_latent = None

        mask_latent = self._resize_mask_to_latent(mask, y_latent) if mask is not None else None

        mask_for_model = mask_latent if self.alg_diffusion_latent_mask else None

        latent_output = self.model.restoration(
            y=y_latent,
            y_cond=x_cond_latent,
            sigmas=sigmas,
            mask=mask_for_model,
            clip_denoised=clip_denoised,
        )

        with torch.no_grad():
            output = self.dc_ae.decode(latent_output).sample

        if mask is not None:
            mask = torch.clamp(mask, min=0.0, max=1.0)
            output = output * mask + (1 - mask) * y

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

    @property
    def cm_model(self):
        return getattr(self.model, "cm_model", None)

    def restoration(
        self,
        y_cond,
        y_t=None,
        y_0=None,
        mask=None,
        sample_num=2,
        cls=None,
        ref=None,
        guidance_scale=0.0,
        ddim_num_steps=10,
        ddim_eta=0.5,
    ):
        self.dc_ae.to(y_cond.device)

        # Encode image-like inputs
        # if y_cond is not None and y_cond.dim() == 4:
        #    y_cond_to_encode = y_cond
        if mask is not None:
            #noise_for_mask = torch.randn_like(y_cond)
            y_cond = y_cond * (1 - mask.float()) + 0.5 * mask.float()
        y_cond_latent = self.dc_ae.encode(y_cond).latent
        # else:
        #    y_cond_latent = y_cond

        # if y_t is not None and y_t.dim() == 4:
        y_t_to_encode = y_t
        if mask is not None:
            #noise_for_mask = torch.randn_like(y_t)
            y_t_to_encode = y_t * (1 - mask.float()) + 0.5 * mask.float()
        y_t_latent = self.dc_ae.encode(y_t_to_encode).latent
        # else:
        #    y_t_latent = y_t

        # Encode image-like inputs
        # if y_0 is not None and y_0.dim() == 4:
        y_0_to_encode = y_0
        if mask is not None:  #
            noise_for_mask = torch.randn_like(y_0)
            y_0_to_encode = y_0 * (1 - mask.float()) + noise_for_mask * mask.float()
        y_0_latent = self.dc_ae.encode(y_0_to_encode).latent
        # else:
        #    y_0_latent = y_0

        if mask is not None and mask.dim() == 4 and self.alg_diffusion_latent_mask:
            # Downsize the mask using nearest neighbors
            downsampled_mask = torch.nn.functional.interpolate(
                mask.float(),
                size=(
                    mask.shape[2] // self.downsampling_factor,
                    mask.shape[3] // self.downsampling_factor,
                ),
                mode="nearest",
            )
            mask_latent = downsampled_mask.repeat(1, self.latent_dim, 1, 1)
            # print('mask latent size=', mask_latent.size())
        else:
            mask_latent = None

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

            # if self.orig_model_output_nc == 1:
            #    output = output.mean(dim=1, keepdim=True)
            #    visuals = visuals.mean(dim=1, keepdim=True)

            ##TODO: reactivate
            if mask is not None and y_cond is not None:
                output = output * mask.float() + y_cond * (1 - mask.float())

        # pixel_space_loss = self.loss_fn(output, y_cond) # debug: since y_cond is the full image
        # latent_space_loss = self.loss_fn(latent_output, y_cond_latent)
        # print('pixel_space_loss=', pixel_space_loss, ' / latent_space_loss=', latent_space_loss)

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
    G_hdit_window_size,
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
    alg_diffusion_latent_dc_ae_scaling=1.0,
    alg_diffusion_finetune_decoder=False,
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
            
        if model_type == "palette":
            if is_pix2pix:
                in_channel = latent_dim * 2
            else:
                in_channel = (
                    latent_dim * 2
                )  # conditioning (akin to input_nc + output_nc in latent space)
        elif model_type == "cm":
            in_channel = latent_dim
            ##TODO: pix2pix case
                
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
        hdit_config = HDiTConfig(
            G_hdit_depths, G_hdit_widths, G_hdit_patch_size, G_hdit_window_size
        )
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
            finetune_decoder=alg_diffusion_finetune_decoder,
            dc_ae_scaling=alg_diffusion_latent_dc_ae_scaling,
        )
        if model_type in {"cm", "cm_gan"} and hasattr(net, "current_t"):
            net.current_t = getattr(net.model, "current_t", net.current_t)

    return net
