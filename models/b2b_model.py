import copy
import hashlib
import itertools
import math
import random
import warnings

import torch
import torchvision.transforms as T
import tqdm
from torch import nn

from util.network_group import NetworkGroup
from util.iter_calculator import IterCalculator
from models.modules.diffusion_utils import rearrange_5dto4d_bf, rearrange_4dto5d_bf

from util.mask_generation import random_edge_mask

from . import diffusion_networks
from .base_diffusion_model import BaseDiffusionModel
from .modules.loss import MultiScaleDiffusionLoss
from piq import DISTS, LPIPS
from torchvision.ops import masks_to_boxes
import torch.nn.functional as F


class B2BModel(BaseDiffusionModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # Base diffusion options
        parser = BaseDiffusionModel.modify_commandline_options(
            parser, is_train=is_train
        )

        # -------------------------
        # B2B diffusion parameters
        # -------------------------

        parser.add_argument(
            "--alg_b2b_minsnr",
            action="store_true",
            help="use min-SNR weighting",
        )
        parser.add_argument(
            "--alg_b2b_autoregressive",
            action="store_true",
            help="Autoregressive training: each batch is with one GT and the other is noisy image ",
        )

        parser.add_argument(
            "--alg_b2b_denoise_timesteps",
            type=int,
            nargs="+",
            default=[50],
            help="One or more denoising step counts to evaluate at inference (positive integers)",
        )
        parser.add_argument(
            "--alg_b2b_noise_scale",
            type=float,
            default=-1.0,
            help="Noise scale for B2B. Use <=0 for automatic JiT-like defaults (1.0 at <=256px, else 2.0).",
        )
        parser.add_argument(
            "--alg_b2b_P_mean",
            type=float,
            default=-0.8,
            help="Mean of the logistic-normal timestep distribution used at B2B training time.",
        )
        parser.add_argument(
            "--alg_b2b_P_std",
            type=float,
            default=0.8,
            help="Std of the logistic-normal timestep distribution used at B2B training time.",
        )
        parser.add_argument(
            "--alg_b2b_t_eps",
            type=float,
            default=5e-2,
            help="Minimum clamp value for (1-t) in velocity conversion v=(x_pred-x)/(1-t).",
        )
        parser.add_argument(
            "--alg_b2b_cfg_scale",
            type=float,
            default=1.0,
            help="Classifier-free guidance scale used at B2B inference time.",
        )
        parser.add_argument(
            "--alg_b2b_clip_denoised",
            action="store_true",
            help="Clip B2B denoised states to [-1, 1] during sampling (disabled by default to match JiT).",
        )
        parser.add_argument(
            "--alg_b2b_disable_inference_clipping",
            action="store_true",
            help="Disable inference-time denominator clipping in v=(x_pred-x)/(1-t), i.e. use raw (1-t) at sampling.",
        )
        parser.add_argument(
            "--alg_b2b_vit_patch_embed_stride_divisor",
            type=int,
            default=1,
            help="Divides the JiT/JiTViD BottleneckPatchEmbed stride by this factor for B2B runs. Use 2 for 50% overlap.",
        )

        # -------------------------
        # Perceptual losses
        # -------------------------
        parser.add_argument(
            "--alg_b2b_perceptual_loss",
            type=str,
            nargs="*",
            default=[""],
            choices=["", "LPIPS", "DISTS"],
            help="Optional perceptual losses",
        )

        parser.add_argument(
            "--alg_b2b_lambda_perceptual",
            type=float,
            default=1.0,
            help="Weight for perceptual loss",
        )

        parser.add_argument(
            "--alg_b2b_dists_mean",
            type=float,
            nargs="*",
            default=[0.485, 0.456, 0.406],
            help="Mean normalization for DISTS",
        )

        parser.add_argument(
            "--alg_b2b_dists_std",
            type=float,
            nargs="*",
            default=[0.229, 0.224, 0.225],
            help="Std normalization for DISTS",
        )

        # -------------------------
        # Evaluation options
        # -------------------------
        parser.add_argument(
            "--alg_b2b_metric_mask",
            action="store_true",
            help="Evaluate metrics only on dilated mask region",
        )

        if is_train:
            parser = B2BModel.modify_commandline_options_train(parser)

        return parser

    @staticmethod
    def modify_commandline_options_train(parser):
        parser = BaseDiffusionModel.modify_commandline_options_train(parser)

        parser.add_argument(
            "--alg_b2b_loss",
            type=str,
            default="MSE",
            choices=["L1", "MSE", "pseudo_huber", "multiscale_L1", "multiscale_MSE"],
            help="Loss type for B2B denoising",
        )
        parser.add_argument(
            "--alg_b2b_loss_masked_region_only",
            action="store_true",
            help="Normalize B2B loss over masked pixels only (instead of all image pixels).",
        )

        return parser

    @staticmethod
    def after_parse(opt):
        steps = getattr(opt, "alg_b2b_denoise_timesteps", [50])
        if isinstance(steps, int):
            steps = [steps]

        if len(steps) == 0:
            raise ValueError(
                "--alg_b2b_denoise_timesteps must contain at least one value"
            )

        if any(step <= 0 for step in steps):
            raise ValueError(
                "--alg_b2b_denoise_timesteps values must be positive integers"
            )

        p_std = getattr(opt, "alg_b2b_P_std", 0.8)
        if p_std <= 0:
            raise ValueError("--alg_b2b_P_std must be > 0")

        t_eps = getattr(opt, "alg_b2b_t_eps", 5e-2)
        if t_eps <= 0:
            raise ValueError("--alg_b2b_t_eps must be > 0")
        if t_eps >= 1:
            raise ValueError("--alg_b2b_t_eps must be < 1")

        patch_stride_divisor = getattr(
            opt, "alg_b2b_vit_patch_embed_stride_divisor", 1
        )
        if patch_stride_divisor < 1:
            raise ValueError(
                "--alg_b2b_vit_patch_embed_stride_divisor must be >= 1"
            )

        opt.alg_b2b_denoise_timesteps = steps
        return opt

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        self.task = self.opt.alg_diffusion_task
        self.gt_frame_idx = 0
        if opt.isTrain:
            batch_size = self.opt.train_batch_size
        else:
            batch_size = self.opt.test_batch_size

        if (
            self.opt.alg_diffusion_cond_embed != ""
            and self.opt.alg_diffusion_generate_per_class
        ):
            self.nb_classes_inference = (
                max(self.opt.f_s_semantic_nclasses, self.opt.cls_semantic_nclasses) - 1
            )

        # Visuals
        visual_outputs = []
        if (
            self.opt.alg_diffusion_cond_embed != ""
            and self.opt.alg_diffusion_generate_per_class
        ):
            for i in range(self.nb_classes_inference):
                self.gen_visual_names.append("output_" + str(i + 1) + "_")
        elif self.opt.model_type in ["b2b", "b2b_gan"]:
            base_names = ["gt_image_", "y_t_", "mask_"]
            step_outputs = [
                f"output_{ts}_steps_" for ts in self.opt.alg_b2b_denoise_timesteps
            ]
            self.gen_visual_names = base_names + step_outputs

        else:
            self.gen_visual_names.append("output_")

        if opt.alg_diffusion_cond_image_creation != "y_t":
            self.gen_visual_names.append("cond_image_")
        if self.opt.alg_diffusion_cond_image_creation == "previous_frame":
            self.gen_visual_names.insert(0, "previous_frame_")

        if self.opt.G_netG == "vit_vid":
            max_visual_outputs = batch_size * self.opt.data_temporal_number_frames
            for k in range(max_visual_outputs):
                self.visual_names.append(
                    [temp + str(k) for temp in self.gen_visual_names]
                )
        else:
            for k in range(self.opt.train_batch_size):
                self.visual_names.append(
                    [temp + str(k) for temp in self.gen_visual_names]
                )
            self.visual_names.append(visual_outputs)

        # Define network
        if getattr(opt, "G_dropout", 0):
            warnings.warn(
                "B2B forces generator dropout to 0. Ignoring non-zero --G_dropout."
            )
        opt.G_dropout = 0.0
        opt.alg_palette_sampling_method = ""
        opt.alg_diffusion_cond_embed = opt.alg_diffusion_cond_image_creation
        opt.alg_diffusion_cond_embed_dim = 256
        self.netG_A = diffusion_networks.define_G(opt=opt, **vars(opt)).to(self.device)
        if opt.isTrain:
            self.netG_A.current_t = max(self.netG_A.current_t, opt.total_iters)
        else:
            self.netG_A.current_t = 0  # placeholder
        print("Setting B2B current_iter to", self.netG_A.current_t)

        self.model_names = ["G_A"]
        self.model_names_export = ["G_A"]

        G_models = ["G_A"]
        G_parameters = [self.netG_A.parameters()]
        G_parameters = itertools.chain(*G_parameters)

        # Define optimizer
        if opt.isTrain:
            self.optimizer_G = opt.optim(
                opt,
                G_parameters,
                lr=opt.train_G_lr,
                betas=(opt.train_beta1, opt.train_beta2),
                weight_decay=opt.train_optim_weight_decay,
                eps=opt.train_optim_eps,
            )
            self.optimizers.append(self.optimizer_G)

        # Define loss functions
        losses_G = ["G_tot"]

        if "multiscale" in self.opt.alg_b2b_loss:
            img_size = self.opt.data_crop_size
            img_size_log = math.floor(math.log2(img_size))
            min_size = 32
            min_size_log = math.floor(math.log2(min_size))

            scales = []
            for k in range(min_size_log, img_size_log + 1):
                scales.append(2**k)
                losses_G.append("G_" + str(2**k))
            losses_G.append("G_" + str(img_size))

            self.loss_fn = MultiScaleDiffusionLoss(
                self.opt.alg_b2b_loss,
                img_size=self.opt.data_crop_size,
                scales=scales,
            )
        elif self.opt.alg_b2b_loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        elif self.opt.alg_b2b_loss == "L1":
            self.loss_fn = torch.nn.L1Loss()
        elif self.opt.alg_b2b_loss == "pseudo_huber":
            self.loss_fn = self._pseudo_huber_loss

        self.loss_names_G = losses_G
        self.loss_names = self.loss_names_G

        # Make group
        self.networks_groups = []

        losses_backward = ["loss_G_tot"]
        self.group_G = NetworkGroup(
            networks_to_optimize=G_models,
            forward_functions=[],
            backward_functions=["compute_b2b_loss"],
            loss_names_list=["loss_names_G"],
            optimizer=["optimizer_G"],
            loss_backward=losses_backward,
            networks_to_ema=G_models,
        )
        self.networks_groups.append(self.group_G)

        losses_G = []
        if opt.alg_b2b_perceptual_loss != [""]:
            losses_G += ["G_perceptual"]
        self.loss_names_G += losses_G
        self.loss_names = self.loss_names_G.copy()

        # Itercalculator
        self.iter_calculator_init()

        # perceptual losses
        if "LPIPS" in self.opt.alg_b2b_perceptual_loss:
            self.criterionLPIPS = LPIPS().to(self.device)
        if "DISTS" in self.opt.alg_b2b_perceptual_loss:
            self.criterionDISTS = DISTS(
                mean=self.opt.alg_b2b_dists_mean, std=self.opt.alg_b2b_dists_std
            ).to(self.device)
        self.requires_x_pred_for_losses = False
        self.pred_x = None

    def set_input(self, data):
        if (
            len(data["A"].to(self.device).shape) == 5
        ) and self.opt.G_netG != "vit_vid":  # we're using temporal successive frames
            self.previous_frame = data["A"].to(self.device)[:, 0]
            self.y_t = data["A"].to(self.device)[:, 1]
            self.gt_image = data["B"].to(self.device)[:, 1]
            self.previous_frame_mask = data["B_label_mask"].to(self.device)[:, 0]
            self.mask = data["B_label_mask"].to(self.device)[:, 1]
        else:
            if self.task == "inpainting":
                # inpainting only
                self.y_t = data["A"].to(self.device)
                self.gt_image = data["B"].to(self.device)
                self.mask = data["B_label_mask"].to(self.device)
            elif self.task == "pix2pix":
                self.y_t = data["A"].to(self.device)
                self.gt_image = data["B"].to(self.device)
                self.mask = None

        if self.opt.alg_diffusion_cond_image_creation == "y_t":
            if self.opt.alg_b2b_autoregressive and self.opt.G_netG == "vit_vid":
                B, T, C, H, W = self.gt_image.shape
                gt_image_mix = self.y_t.clone()
                self.cond_image = None

                # per-sample decision: True for ~10% of samples in the batch
                use_gt = torch.rand((B,), device=self.device) < 0.1
                # one random frame index per sample (we'll only use those for sel)
                idx = torch.randint(0, T, (B,), device=self.device)
                self.use_gt = use_gt  # (B,) bool
                self.ref_idx = idx  # (B,) long, valid even if use_gt False
                if use_gt.any():
                    batch_idx = torch.arange(B, device=self.device)
                    sel = batch_idx[use_gt]  # selected samples only

                    # replace y_t frame by GT for selected samples
                    gt_image_mix[sel, idx[use_gt]] = self.gt_image[sel, idx[use_gt]]
                    self.y_t = gt_image_mix

                    # mask: keep that GT frame clean (0), diffuse the rest (1)
                    mask_ar = torch.ones((B, T, 1, 1, 1), device=self.device)
                    mask_ar[sel, idx[use_gt]] = 0.0
                    self.mask = self.mask * mask_ar

                # else: nobody selected -> do nothing
            else:
                self.cond_image = None

        self.batch_size = self.y_t.shape[0]
        self.image_paths = data.get("A_img_paths", [])

        self.real_A = self.y_t
        self.real_B = self.gt_image
        self.num_classes = getattr(self.opt, "G_vit_num_classes", 1) if self.opt else 1

        if self.num_classes != 1:
            raise RuntimeError(
                f"Expected G_vit_num_classes == 1, but got {self.num_classes}. "
                "Stopping because this run only supports num_classes=1."
            )
        self.label_cls = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )

    def compute_b2b_loss(self):
        y_0 = self.gt_image
        y_cond = self.cond_image
        mask = self.mask
        B = y_0.shape[0]
        use_perceptual = self.opt.alg_b2b_perceptual_loss != [""]
        need_x_pred = use_perceptual or self.requires_x_pred_for_losses

        # base "real" labels in [0 .. num_classes-1]
        if self.num_classes > 0:
            labels = torch.zeros(
                B, dtype=torch.long, device=self.device
            )  # class 1 start from 0, need discuss
        else:
            labels = None

        net_out = self.netG_A(
            y_0,
            mask,
            y_cond,
            label=labels,
            use_gt=getattr(self, "use_gt", None),
            ref_idx=getattr(self, "ref_idx", None),
            return_x_pred=need_x_pred,
        )
        if need_x_pred:
            v_pred, v, x_pred = net_out
        else:
            v_pred, v = net_out
            x_pred = None
        self.pred_x = x_pred
        if x_pred is not None:
            self.fake_B = x_pred

        if not self.opt.alg_b2b_minsnr:
            min_snr_loss_weight = 1.0

        if mask is not None:
            mask_binary = torch.clamp(mask, min=0, max=1)
            weighted_pred = min_snr_loss_weight * v_pred
            weighted_target = min_snr_loss_weight * v
            if self.opt.alg_b2b_loss_masked_region_only:
                if isinstance(self.loss_fn, MultiScaleDiffusionLoss):
                    loss = self.loss_fn(
                        weighted_pred,
                        weighted_target,
                        mask=mask_binary,
                        masked_region_only=True,
                    )
                else:
                    loss = self._masked_region_loss(
                        weighted_pred, weighted_target, mask_binary
                    )
            else:
                loss = self.loss_fn(
                    mask_binary * weighted_pred,
                    mask_binary * weighted_target,
                )
        else:
            loss = self.loss_fn(min_snr_loss_weight * v_pred, min_snr_loss_weight * v)

        if isinstance(loss, dict):
            loss_tot = torch.zeros(size=(), device=y_0.device)

            for cur_size, cur_loss in loss.items():
                setattr(self, "loss_G_" + cur_size, cur_loss)
                loss_tot += cur_loss

            loss = loss_tot

        self.loss_G_tot = self.opt.alg_diffusion_lambda_G * loss

        self.loss_G_perceptual_lpips = torch.zeros(size=(), device=y_0.device)
        self.loss_G_perceptual_dists = torch.zeros(size=(), device=y_0.device)
        self.loss_G_perceptual = torch.zeros(size=(), device=y_0.device)

        if use_perceptual and x_pred is not None:
            if mask is not None:
                mask_binary = torch.clamp(mask, min=0, max=1)
                target_for_perceptual = y_0 * mask_binary
                pred_for_perceptual = x_pred * mask_binary
            else:
                target_for_perceptual = y_0
                pred_for_perceptual = x_pred

            (
                self.loss_G_perceptual_lpips,
                self.loss_G_perceptual_dists,
            ) = self._compute_perceptual_losses(
                target_for_perceptual, pred_for_perceptual
            )
            self.loss_G_perceptual = self.opt.alg_b2b_lambda_perceptual * (
                self.loss_G_perceptual_lpips + self.loss_G_perceptual_dists
            )
            self.loss_G_tot += self.loss_G_perceptual

    def _masked_region_loss(self, pred, target, mask, eps=1e-8):
        if self.opt.alg_b2b_loss in ["MSE", "multiscale_MSE"]:
            loss_elem = (pred - target) ** 2
        elif self.opt.alg_b2b_loss in ["L1", "multiscale_L1"]:
            loss_elem = torch.abs(pred - target)
        elif self.opt.alg_b2b_loss == "pseudo_huber":
            c = 0.00054 * math.sqrt(math.prod(pred.shape[1:]))
            loss_elem = torch.sqrt((pred - target) ** 2 + c**2) - c
        else:
            raise NotImplementedError(
                f"Unsupported alg_b2b_loss for masked-region loss: {self.opt.alg_b2b_loss}"
            )

        reduce_dims = tuple(range(1, loss_elem.ndim))
        num = (loss_elem * mask).sum(dim=reduce_dims)
        den = mask.sum(dim=reduce_dims).clamp_min(eps)
        return (num / den).mean()

    def _pseudo_huber_loss(self, pred, target):
        c = 0.00054 * math.sqrt(math.prod(pred.shape[1:]))
        return torch.mean(torch.sqrt((pred - target) ** 2 + c**2) - c)

    def _compute_perceptual_losses(self, target, pred):
        if target.ndim == 5:
            target, pred = rearrange_5dto4d_bf(target, pred)

        lpips_loss = torch.zeros(size=(), device=target.device)
        dists_loss = torch.zeros(size=(), device=target.device)

        if "LPIPS" in self.opt.alg_b2b_perceptual_loss:
            lpips_loss = self._perceptual_loss_by_channel(
                self.criterionLPIPS, target, pred
            )
        if "DISTS" in self.opt.alg_b2b_perceptual_loss:
            dists_loss = self._perceptual_loss_by_channel(
                self.criterionDISTS, target, pred
            )

        return lpips_loss, dists_loss

    def _perceptual_loss_by_channel(self, criterion, target, pred):
        if pred.size(1) > 3:
            loss = torch.zeros(size=(), device=pred.device)
            for c in range(pred.size(1)):
                loss += torch.mean(
                    criterion(
                        target[:, c : c + 1, :, :],
                        pred[:, c : c + 1, :, :],
                    )
                )
            return loss
        return torch.mean(criterion(target, pred))

    def inference(self, nb_imgs, offset=0):
        offset = 0
        if (
            not self.opt.isTrain
            and getattr(self.opt, "train_G_ema", False)
            and hasattr(self, "netG_A_ema")
        ):
            netG = self.netG_A_ema
        else:
            netG = self.netG_A

        if hasattr(netG, "module"):
            netG = netG.module

        if len(self.opt.gpu_ids) > 1 and self.opt.G_unet_mha_norm_layer == "batchnorm":
            netG = revert_sync_batchnorm(netG)

        # XXX: inpainting only for now
        if self.mask is not None:
            mask = self.mask[:nb_imgs]
        else:
            mask = self.mask
        # restoration call
        if not self.cond_image is None:
            y_cond = self.cond_image[:nb_imgs]
        else:
            y_cond = None
        if self.task == "pix2pix":  #
            out_shape = list(y_cond.shape)
            out_shape[1] = netG.b2b_model.out_channel
            y_t = torch.zeros(out_shape, device=y_cond.device, dtype=y_cond.dtype)
        else:  # e.g. inpainting
            y_t = self.y_t[:nb_imgs]

        if self.task in ["inpainting"]:
            if (
                self.opt.alg_diffusion_cond_embed != ""
                and self.opt.alg_diffusion_generate_per_class
            ):

                for i in range(self.nb_classes_inference):
                    if "mask" in self.opt.alg_diffusion_cond_embed:
                        cur_mask = self.mask[:nb_imgs].clone().clamp(min=0, max=1) * (
                            i + 1
                        )
                    else:
                        cur_mask = self.mask[:nb_imgs]
                    self.output = netG.restoration(
                        y_t, y_cond, self.opt.alg_b2b_denoise_timesteps, cur_mask
                    )
                    name = "output_" + str(i + 1)
                    setattr(self, name, self.output)
            else:
                self.outputs_per_step = {}
                self.fake_B_dilated_per_step = {}
                self.gt_image_dilated_per_step = {}
                use_gt = getattr(self, "use_gt", None)
                ref_idx = getattr(self, "ref_idx", None)
                shared_init_noise = None
                if (
                    not self.opt.isTrain
                    and isinstance(self.opt.alg_b2b_denoise_timesteps, list)
                    and len(self.opt.alg_b2b_denoise_timesteps) > 1
                ):
                    shared_init_noise = self._build_fixed_validation_noise(y_t)
                for steps in self.opt.alg_b2b_denoise_timesteps:
                    self.output = netG.restoration(
                        y_t,
                        y_cond,
                        steps,
                        mask,
                        self.label_cls,
                        use_gt=use_gt,
                        ref_idx=ref_idx,
                        init_noise=shared_init_noise,
                    )
                    self.outputs_per_step[steps] = self.output
                    name = "output_" + str(steps) + "_steps"
                    setattr(self, name, self.output)

        num_steps = len(self.outputs_per_step)
        if num_steps > 0:
            self.fake_B = torch.cat(list(self.outputs_per_step.values()), dim=0)
            if num_steps > 1:
                repeat_dims = [num_steps] + [1] * (self.gt_image.dim() - 1)
                self.gt_image = self.gt_image.repeat(*repeat_dims)
        else:
            self.fake_B = self.output

        self.visuals = self.fake_B  # self.output

        if not self.opt.G_netG == "vit_vid":
            for name in self.gen_visual_names:
                whole_tensor = getattr(self, name[:-1])  # i.e. self.output, ...
                for k in range(min(nb_imgs, self.get_current_batch_size())):
                    cur_name = name + str(offset + k)
                    cur_tensor = whole_tensor[k : k + 1]
                    if "mask" in name:
                        cur_tensor = cur_tensor.squeeze(0)
                    setattr(self, cur_name, cur_tensor)
        else:
            for name in self.gen_visual_names:
                whole_tensor = getattr(self, name[:-1])  # i.e. self.output, ...
                for bs in range(min(nb_imgs, self.get_current_batch_size())):
                    for k in range(self.opt.data_temporal_number_frames):
                        cur_name = name + str(
                            offset + bs * (self.opt.data_temporal_number_frames) + k
                        )
                        cur_tensor = whole_tensor[bs, k, :, :, :].unsqueeze(0)
                        if "mask" in name:
                            cur_tensor = cur_tensor.squeeze(0)
                        setattr(self, cur_name, cur_tensor)

    def compute_visuals(self, nb_imgs):
        super().compute_visuals(nb_imgs)
        with torch.no_grad():
            self.inference(nb_imgs)

    def get_current_visuals(self, nb_imgs, phase="train", test_name=""):
        # hide noisy columns in test mode
        old_visual_names = self.visual_names.copy()
        if phase == "test":
            self.visual_names = [
                [x for x in visual_name if "noisy" not in x]
                for visual_name in self.visual_names
            ]
        x = super().get_current_visuals(nb_imgs, phase, test_name)
        self.visual_names = old_visual_names
        return x

    def _normalize_image_paths_for_batch(self, batch_size):
        paths = getattr(self, "image_paths", [])

        if isinstance(paths, str):
            paths_list = [paths]
        elif isinstance(paths, (list, tuple)):
            paths_list = list(paths)
        else:
            paths_list = []

        if len(paths_list) == batch_size:
            return [str(p) for p in paths_list]

        if len(paths_list) == 1 and batch_size > 1:
            base = str(paths_list[0])
            return [f"{base}#b{i}" for i in range(batch_size)]

        return [f"batch_item_{i}" for i in range(batch_size)]

    def _stable_seed_from_path(self, path):
        digest = hashlib.sha256(path.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    def _build_fixed_validation_noise(self, y):
        batch_size = y.shape[0]
        paths = self._normalize_image_paths_for_batch(batch_size)

        per_sample_noise = []
        for i, path in enumerate(paths):
            seed = self._stable_seed_from_path(path)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            sample_noise = torch.randn(
                tuple(y[i].shape),
                generator=gen,
                device="cpu",
                dtype=y.dtype,
            )
            per_sample_noise.append(sample_noise)

        noise = torch.stack(per_sample_noise, dim=0).to(device=y.device, dtype=y.dtype)
        return noise
