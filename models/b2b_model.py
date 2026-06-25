import copy
import hashlib
import math
import os
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
from data.base_dataset import transform_object_reference_images
from util.b2b_context import (
    VALID_B2B_GLOBAL_CONTEXT_MODES,
    b2b_global_context_enabled,
    b2b_global_context_mode_from_opt,
    b2b_global_context_tokens_enabled,
)


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
            "--alg_b2b_use_gt_prob",
            type=float,
            default=0.1,
            help="Probability of selecting a sample to use a GT frame in autoregressive B2B training.",
        )
        parser.add_argument(
            "--alg_b2b_mask_as_channel",
            action="store_true",
            help="Concatenate the inpainting mask as an additional input channel in B2B.",
        )
        parser.add_argument(
            "--alg_b2b_mask_size_conditioning",
            action="store_true",
            help=(
                "Condition JiT/JiTViD B2B denoisers on normalized mask bbox "
                "geometry (center, size, area, aspect)."
            ),
        )
        parser.add_argument(
            "--alg_b2b_temporal_frame_step_conditioning",
            action="store_true",
            help=(
                "Condition JiTViD B2B denoisers on the raw temporal frame stride "
                "used to build each video sample."
            ),
        )
        parser.add_argument(
            "--alg_b2b_global_context_conditioning",
            action="store_true",
            help=(
                "Condition JiTViD B2B denoisers on masked full-frame context "
                "encoded by a small CNN. Deprecated compatibility alias for "
                "--alg_b2b_global_context_mode adaln."
            ),
        )
        parser.add_argument(
            "--alg_b2b_global_context_mode",
            choices=VALID_B2B_GLOBAL_CONTEXT_MODES,
            default="none",
            help=(
                "Global context conditioning mode for JiTViD B2B. "
                "'adaln' uses the legacy small-CNN AdaLN path, 'tokens' inserts "
                "global context ViT prefix tokens, and 'both' enables both paths."
            ),
        )
        parser.add_argument(
            "--alg_b2b_global_context_size",
            type=int,
            default=128,
            help="Square input size for the masked full-frame B2B context encoder.",
        )
        parser.add_argument(
            "--alg_b2b_object_ref_paths",
            type=str,
            nargs="*",
            default=[],
            help=(
                "Static object reference image paths. Providing one or more paths "
                "enables object-reference token conditioning for vit_vid B2B."
            ),
        )
        parser.add_argument(
            "--alg_b2b_object_ref_size",
            type=int,
            default=64,
            help=(
                "Square padded input size for object-reference token conditioning. "
                "Must be divisible by the ViT patch size."
            ),
        )
        parser.add_argument(
            "--alg_b2b_multi_dataset_class_conditioning",
            action="store_true",
            help=(
                "Use multi_dataset dataset_index as the ViT class-token "
                "conditioning label instead of object class labels."
            ),
        )
        parser.add_argument(
            "--alg_b2b_force_class_token",
            type=int,
            default=-1,
            help=(
                "Force every B2B sample to use this ViT class-token label. "
                "Use -1 to keep dataset/object-label behavior."
            ),
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
            "--alg_b2b_timestep_uniform_mix_prob",
            type=float,
            default=0.1,
            help=(
                "Probability of replacing a logistic-normal B2B training timestep "
                "with a uniform sample in [0, 1]."
            ),
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
        parser.add_argument(
            "--alg_b2b_lambda_ref_copy",
            type=float,
            default=0.0,
            help=(
                "Weight for an image-space copy loss on autoregressive B2B "
                "reference frames before mask projection. 0 disables it."
            ),
        )
        parser.add_argument(
            "--alg_b2b_ref_degrade_prob",
            type=float,
            default=0.0,
            help=(
                "Probability of adding Gaussian noise to selected autoregressive "
                "reference frames in the model input."
            ),
        )
        parser.add_argument(
            "--alg_b2b_ref_degrade_noise_std",
            type=float,
            default=0.05,
            help="Gaussian noise std for degraded autoregressive reference frames.",
        )
        parser.add_argument(
            "--alg_b2b_lora",
            action="store_true",
            help=(
                "Train B2B JiT/JiTViD with PEFT LoRA adapters while saving merged "
                "full checkpoints."
            ),
        )
        parser.add_argument(
            "--alg_b2b_lora_rank",
            type=int,
            default=8,
            help="LoRA rank for B2B JiT/JiTViD finetuning.",
        )
        parser.add_argument(
            "--alg_b2b_lora_alpha",
            type=int,
            default=16,
            help="LoRA alpha scaling for B2B JiT/JiTViD finetuning.",
        )
        parser.add_argument(
            "--alg_b2b_lora_dropout",
            type=float,
            default=0.05,
            help="LoRA dropout for B2B JiT/JiTViD finetuning.",
        )
        parser.add_argument(
            "--alg_b2b_lora_target_modules",
            type=str,
            nargs="+",
            default=["attn.qkv", "attn.proj", "mlp.w12", "mlp.w3"],
            help=(
                "Module suffixes targeted by B2B LoRA. Defaults to attention and "
                "MLP projections in JiT/JiTViD blocks."
            ),
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

        uniform_mix_prob = getattr(opt, "alg_b2b_timestep_uniform_mix_prob", 0.1)
        if not (0.0 <= uniform_mix_prob <= 1.0):
            raise ValueError("--alg_b2b_timestep_uniform_mix_prob must be in [0, 1]")

        t_eps = getattr(opt, "alg_b2b_t_eps", 5e-2)
        if t_eps <= 0:
            raise ValueError("--alg_b2b_t_eps must be > 0")
        if t_eps >= 1:
            raise ValueError("--alg_b2b_t_eps must be < 1")

        use_gt_prob = getattr(opt, "alg_b2b_use_gt_prob", 0.1)
        if not (0.0 <= use_gt_prob <= 1.0):
            raise ValueError("--alg_b2b_use_gt_prob must be in [0, 1]")

        if (
            getattr(opt, "alg_b2b_multi_dataset_class_conditioning", False)
            and getattr(opt, "data_dataset_mode", "") != "multi_dataset"
        ):
            raise ValueError(
                "--alg_b2b_multi_dataset_class_conditioning requires "
                "--data_dataset_mode multi_dataset"
            )

        if getattr(opt, "alg_b2b_mask_size_conditioning", False) and getattr(
            opt, "G_netG", ""
        ) not in ["vit", "vit_vid"]:
            raise ValueError(
                "--alg_b2b_mask_size_conditioning is only supported with vit/vit_vid B2B"
            )
        forced_class_token = int(getattr(opt, "alg_b2b_force_class_token", -1))
        if forced_class_token < -1:
            raise ValueError("--alg_b2b_force_class_token must be -1 or >= 0")
        if forced_class_token >= 0:
            if getattr(opt, "G_netG", "") not in ["vit", "vit_vid"]:
                raise ValueError(
                    "--alg_b2b_force_class_token is only supported with vit/vit_vid B2B"
                )
            num_classes = int(getattr(opt, "G_vit_num_classes", 1))
            if forced_class_token >= num_classes:
                raise ValueError(
                    "--alg_b2b_force_class_token must be < --G_vit_num_classes"
                )
        if getattr(opt, "alg_b2b_temporal_frame_step_conditioning", False):
            if getattr(opt, "G_netG", "") != "vit_vid":
                raise ValueError(
                    "--alg_b2b_temporal_frame_step_conditioning is only supported "
                    "with vit_vid B2B"
                )
            if int(getattr(opt, "data_temporal_number_frames", 1)) < 2:
                raise ValueError(
                    "--alg_b2b_temporal_frame_step_conditioning requires "
                    "--data_temporal_number_frames >= 2"
                )
        global_context_mode = b2b_global_context_mode_from_opt(opt)
        opt.alg_b2b_global_context_mode = global_context_mode
        opt.alg_b2b_global_context_conditioning = b2b_global_context_enabled(
            global_context_mode
        )
        if b2b_global_context_enabled(global_context_mode):
            if getattr(opt, "G_netG", "") != "vit_vid":
                raise ValueError(
                    "--alg_b2b_global_context_mode/"
                    "--alg_b2b_global_context_conditioning is only supported "
                    "with vit_vid B2B"
                )
            if getattr(opt, "alg_b2b_global_context_size", 128) <= 0:
                raise ValueError("--alg_b2b_global_context_size must be > 0")
            if b2b_global_context_tokens_enabled(global_context_mode):
                patch_size = int(getattr(opt, "G_vit_patch_size", 16))
                if int(getattr(opt, "alg_b2b_global_context_size", 128)) % patch_size:
                    raise ValueError(
                        "--alg_b2b_global_context_size must be divisible by "
                        "--G_vit_patch_size for token global context conditioning"
                    )
            if getattr(opt, "dataaug_affine", 0):
                raise ValueError(
                    "--alg_b2b_global_context_mode does not support dataaug_affine"
                )

        object_ref_paths = getattr(opt, "alg_b2b_object_ref_paths", []) or []
        if object_ref_paths:
            if getattr(opt, "G_netG", "") != "vit_vid":
                raise ValueError(
                    "--alg_b2b_object_ref_paths is only supported with vit_vid B2B"
                )
            object_ref_size = int(getattr(opt, "alg_b2b_object_ref_size", 64))
            if object_ref_size <= 0:
                raise ValueError("--alg_b2b_object_ref_size must be > 0")
            patch_size = int(getattr(opt, "G_vit_patch_size", 16))
            if object_ref_size % patch_size != 0:
                raise ValueError(
                    "--alg_b2b_object_ref_size must be divisible by --G_vit_patch_size"
                )

        if getattr(opt, "G_vit_vid_motion_every", 0) < 0:
            raise ValueError("--G_vit_vid_motion_every must be >= 0")

        if getattr(opt, "alg_b2b_lambda_ref_copy", 0.0) < 0:
            raise ValueError("--alg_b2b_lambda_ref_copy must be >= 0")
        ref_degrade_prob = getattr(opt, "alg_b2b_ref_degrade_prob", 0.0)
        if not (0.0 <= ref_degrade_prob <= 1.0):
            raise ValueError("--alg_b2b_ref_degrade_prob must be in [0, 1]")
        if getattr(opt, "alg_b2b_ref_degrade_noise_std", 0.05) < 0:
            raise ValueError("--alg_b2b_ref_degrade_noise_std must be >= 0")

        if getattr(opt, "alg_b2b_lora", False) and getattr(opt, "isTrain", False):
            if getattr(opt, "G_netG", "") not in ["vit", "vit_vid"]:
                raise ValueError(
                    "--alg_b2b_lora is only supported with vit/vit_vid B2B"
                )
            if getattr(opt, "alg_b2b_lora_rank", 8) <= 0:
                raise ValueError("--alg_b2b_lora_rank must be > 0")
            if getattr(opt, "alg_b2b_lora_alpha", 16) <= 0:
                raise ValueError("--alg_b2b_lora_alpha must be > 0")
            lora_dropout = getattr(opt, "alg_b2b_lora_dropout", 0.05)
            if not (0.0 <= lora_dropout < 1.0):
                raise ValueError("--alg_b2b_lora_dropout must be in [0, 1)")
            lora_targets = getattr(opt, "alg_b2b_lora_target_modules", [])
            if not lora_targets or any(not target for target in lora_targets):
                raise ValueError(
                    "--alg_b2b_lora_target_modules must contain at least one module"
                )

        opt.alg_b2b_denoise_timesteps = steps
        return opt

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        self.task = self.opt.alg_diffusion_task

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
        elif self.opt.model_type == "b2b":
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
        self.object_refs = self._load_b2b_object_references()
        self._apply_b2b_lora()
        if opt.isTrain:
            self.netG_A.current_t = max(self.netG_A.current_t, opt.total_iters)
        else:
            self.netG_A.current_t = 0  # placeholder
        print("Setting B2B current_iter to", self.netG_A.current_t)

        self.model_names = ["G_A"]
        self.model_names_export = ["G_A"]

        G_models = ["G_A"]
        G_parameters = self._iter_trainable_generator_parameters()

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
        if getattr(opt, "alg_b2b_lambda_ref_copy", 0.0) > 0:
            losses_G += ["G_ref_copy"]
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

    def _should_apply_b2b_lora(self):
        return bool(getattr(self.opt, "alg_b2b_lora", False)) and bool(
            getattr(self.opt, "isTrain", False)
        )

    def _apply_b2b_lora(self):
        if not self._should_apply_b2b_lora():
            return
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "--alg_b2b_lora requires the peft package to be installed"
            ) from exc

        config = LoraConfig(
            r=self.opt.alg_b2b_lora_rank,
            lora_alpha=self.opt.alg_b2b_lora_alpha,
            lora_dropout=self.opt.alg_b2b_lora_dropout,
            target_modules=list(self.opt.alg_b2b_lora_target_modules),
            bias="none",
        )
        self.netG_A.b2b_model = get_peft_model(self.netG_A.b2b_model, config)
        self.netG_A.b2b_lora_enabled = True
        self._set_b2b_lora_trainability(self.netG_A)

    def _unwrap_net(self, net):
        if hasattr(net, "module"):
            return net.module
        return net

    def _set_b2b_lora_trainability(self, net=None):
        if not self._should_apply_b2b_lora():
            return
        if net is None:
            net = self.netG_A
        net = self._unwrap_net(net)
        for param in net.parameters():
            param.requires_grad = False
        trainable = []
        for name, param in net.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                trainable.append(name)
        if not trainable:
            raise RuntimeError(
                "B2B LoRA is enabled but no LoRA parameters were found on netG_A"
            )

    def _iter_trainable_generator_parameters(self):
        if self._should_apply_b2b_lora():
            self._set_b2b_lora_trainability(self.netG_A)
        params = [param for param in self.netG_A.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError("No trainable B2B generator parameters found")
        return params

    def parallelize(self, rank):
        if not self._should_apply_b2b_lora():
            super().parallelize(rank)
            return

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name).to(self.gpu_ids[rank])
                self._set_b2b_lora_trainability(net)
                net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
                self._set_b2b_lora_trainability(net)
                setattr(
                    self,
                    "net" + name,
                    torch.nn.parallel.DistributedDataParallel(
                        net, device_ids=[self.gpu_ids[rank]], broadcast_buffers=False
                    ),
                )

    def single_gpu(self):
        super().single_gpu()
        self._set_b2b_lora_trainability(self.netG_A)

    def _b2b_lora_target_modules(self):
        return list(getattr(self.opt, "alg_b2b_lora_target_modules", []))

    def _is_b2b_lora_target(self, module_path):
        return any(
            module_path.endswith(target) for target in self._b2b_lora_target_modules()
        )

    def _map_raw_b2b_key_to_lora_key(self, key):
        prefix = "b2b_model."
        if not key.startswith(prefix):
            return key

        rest = key[len(prefix) :]
        for suffix in [".weight", ".bias"]:
            if rest.endswith(suffix):
                module_path = rest[: -len(suffix)]
                if self._is_b2b_lora_target(module_path):
                    return (
                        "b2b_model.base_model.model."
                        + module_path
                        + ".base_layer"
                        + suffix
                    )
        return "b2b_model.base_model.model." + rest

    def _checkpoint_has_b2b_lora_keys(self, state_dict):
        return any(
            "base_model.model" in key or "base_layer" in key or "lora_" in key
            for key in state_dict.keys()
        )

    def _load_raw_b2b_checkpoint_into_lora(self, net, state_dict, name):
        model_dict = net.state_dict()
        mapped_state = {}
        skipped = []
        for key, value in state_dict.items():
            new_key = self._map_raw_b2b_key_to_lora_key(key)
            if new_key in model_dict and value.shape == model_dict[new_key].shape:
                mapped_state[new_key] = value
            else:
                skipped.append((key, new_key))

        self._adjust_positional_embeddings(mapped_state, model_dict, name)
        missing = set(model_dict.keys()) - set(mapped_state.keys())
        non_lora_missing = [key for key in missing if "lora_" not in key]

        print(
            f"\n===== B2B LoRA raw checkpoint load report for {name} =====",
            flush=True,
        )
        print(f"Model params keys: {len(model_dict)}")
        print(f"Checkpoint keys:   {len(state_dict)}")
        print(f"Mapped keys:       {len(mapped_state)}")
        print(f"Missing LoRA keys: {len(missing) - len(non_lora_missing)}")
        print(f"Missing base keys: {len(non_lora_missing)}")
        print(f"Skipped ckpt keys: {len(skipped)}")
        print("=====================================================\n")

        if self.opt.model_load_no_strictness and non_lora_missing:
            preview = "\n".join(sorted(non_lora_missing)[:20])
            raise RuntimeError(
                "Missing non-LoRA keys while loading raw B2B checkpoint into LoRA "
                f"model:\n{preview}"
            )
        net.load_state_dict(mapped_state, strict=False)

    def _merged_b2b_lora_state_dict(self, net):
        net = self._unwrap_net(net)
        net_to_save = copy.deepcopy(net).to("cpu")
        if hasattr(net_to_save.b2b_model, "merge_and_unload"):
            net_to_save.b2b_model = net_to_save.b2b_model.merge_and_unload()
        return net_to_save.state_dict()

    def save_networks(self, epoch):
        if not self._should_apply_b2b_lora():
            super().save_networks(epoch)
            return

        for name in self.model_names:
            if not isinstance(name, str):
                continue
            save_filename = "%s_net_%s.pth" % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, "net" + name)
            torch.save(self._merged_b2b_lora_state_dict(net), save_path)

            if self.opt.train_G_ema:
                net_ema = getattr(self, "net" + name + "_ema", None)
                if net_ema is not None:
                    ema_save_filename = "%s_net_%s_ema.pth" % (epoch, name)
                    ema_save_path = os.path.join(self.save_dir, ema_save_filename)
                    torch.save(self._merged_b2b_lora_state_dict(net_ema), ema_save_path)

    def load_networks(self, epoch, load_dir=None):
        if not self._should_apply_b2b_lora():
            super().load_networks(epoch, load_dir=load_dir)
            return

        if load_dir is None:
            load_dir = self.save_dir
        for name in self.model_names:
            if not isinstance(name, str):
                continue

            load_filename = "%s_net_%s.pth" % (epoch, name)
            load_path = os.path.join(load_dir, load_filename)
            print("loading the model from %s" % load_path)

            net = self._unwrap_net(getattr(self, "net" + name))
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, "_metadata"):
                del state_dict._metadata
            if isinstance(state_dict, dict) and "_ema" in state_dict:
                state_dict = state_dict["_ema"]

            if self._checkpoint_has_b2b_lora_keys(state_dict):
                net.load_state_dict(
                    state_dict, strict=self.opt.model_load_no_strictness
                )
                continue

            self._load_raw_b2b_checkpoint_into_lora(net, state_dict, name)

    def _b2b_global_context_from_data(self, data):
        if not b2b_global_context_enabled(b2b_global_context_mode_from_opt(self.opt)):
            return None
        global_context = data.get("B_global_context", data.get("A_global_context"))
        if global_context is None:
            if getattr(self.opt, "isTrain", False):
                raise RuntimeError(
                    "--alg_b2b_global_context_conditioning requires "
                    "B_global_context or A_global_context from the dataset"
                )
            return None
        return global_context.to(self.device)

    def _b2b_temporal_frame_step_from_data(self, data):
        if not getattr(self.opt, "alg_b2b_temporal_frame_step_conditioning", False):
            return None

        frame_step = data.get(
            "temporal_frame_step",
            data.get("B_temporal_frame_step", data.get("A_temporal_frame_step")),
        )
        if frame_step is None:
            if getattr(self.opt, "isTrain", False):
                raise RuntimeError(
                    "--alg_b2b_temporal_frame_step_conditioning requires "
                    "temporal_frame_step from the dataset"
                )
            frame_step = torch.full(
                (self.batch_size,),
                int(getattr(self.opt, "data_temporal_frame_step", 1)),
                device=self.device,
                dtype=torch.float32,
            )
        elif not torch.is_tensor(frame_step):
            frame_step = torch.as_tensor(frame_step)

        frame_step = frame_step.to(device=self.device, dtype=torch.float32)
        if frame_step.ndim == 0:
            frame_step = frame_step.reshape(1).expand(self.batch_size)
        return frame_step

    def _load_b2b_object_references(self):
        paths = getattr(self.opt, "alg_b2b_object_ref_paths", []) or []
        if not paths:
            return None
        refs = transform_object_reference_images(
            list(paths), int(getattr(self.opt, "alg_b2b_object_ref_size", 64))
        )
        return refs.to(self.device)

    def set_input(self, data):
        self.global_context = self._b2b_global_context_from_data(data)
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

                # per-sample decision: True for ~P of samples in the batch
                use_gt = (
                    torch.rand((B,), device=self.device) < self.opt.alg_b2b_use_gt_prob
                )
                # one random frame index per sample (we'll only use those for sel)
                idx = torch.randint(0, T, (B,), device=self.device)
                self.use_gt = use_gt  # (B,) bool
                self.ref_idx = idx  # (B,) long, valid even if use_gt False
                if use_gt.any():
                    batch_idx = torch.arange(B, device=self.device)
                    sel = batch_idx[use_gt]  # selected samples only

                    # replace y_t frame by GT for selected samples
                    gt_image_mix[sel, idx[use_gt]] = self.gt_image[sel, idx[use_gt]]
                    gt_image_mix = self._degrade_b2b_reference_frames(
                        gt_image_mix, use_gt, idx
                    )
                    self.y_t = gt_image_mix

                    # mask: keep that GT frame clean (0), diffuse the rest (1)
                    mask_ar = torch.ones((B, T, 1, 1, 1), device=self.device)
                    mask_ar[sel, idx[use_gt]] = 0.0
                    self.mask = self.mask * mask_ar

                # else: nobody selected -> do nothing
        self._apply_b2b_diff_augment()
        if self.opt.alg_diffusion_cond_image_creation == "y_t":
            if self.opt.alg_b2b_mask_as_channel:
                if self.mask is None:
                    raise RuntimeError(
                        "--alg_b2b_mask_as_channel requires inpainting masks."
                    )
                self.cond_image = self.mask.to(dtype=self.y_t.dtype)
            else:
                self.cond_image = None

        self.batch_size = self.y_t.shape[0]
        self.temporal_frame_step = self._b2b_temporal_frame_step_from_data(data)
        self.image_paths = data.get("A_img_paths", [])

        self.real_A = self.y_t
        self.real_B = self.gt_image
        self.num_classes = getattr(self.opt, "G_vit_num_classes", 1) if self.opt else 1

        self.label_cls = self._select_b2b_labels(data)

    def _default_b2b_labels(self):
        return torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

    def _select_b2b_labels(self, data):
        forced_class_token = int(getattr(self.opt, "alg_b2b_force_class_token", -1))
        if forced_class_token >= 0:
            return torch.full(
                (self.batch_size,),
                forced_class_token,
                dtype=torch.long,
                device=self.device,
            )

        if getattr(self.opt, "alg_b2b_multi_dataset_class_conditioning", False):
            return self._prepare_b2b_dataset_labels(data.get("dataset_index"))

        label_cls = data.get("B_label_cls", data.get("A_label_cls"))
        return self._prepare_b2b_labels(label_cls)

    def _prepare_b2b_labels(self, labels):
        if labels is None:
            return self._default_b2b_labels()

        labels = labels.to(torch.long).to(self.device)
        is_video = self.y_t.ndim == 5
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        if labels.ndim == 1:
            if labels.shape[0] != self.batch_size:
                raise RuntimeError(
                    f"Expected class labels batch dim {self.batch_size}, got {tuple(labels.shape)}"
                )
            return labels
        if labels.ndim == 2 and is_video:
            if labels.shape[0] != self.batch_size:
                raise RuntimeError(
                    f"Expected video class labels batch dim {self.batch_size}, got {tuple(labels.shape)}"
                )
            if labels.shape[1] != self.y_t.shape[1]:
                raise RuntimeError(
                    f"Expected per-frame class labels with {self.y_t.shape[1]} frames, got {tuple(labels.shape)}"
                )
            return labels
        if labels.ndim != 1:
            raise RuntimeError(
                f"Expected class labels with shape [B] or [B, F] for video, got {tuple(labels.shape)}"
            )
        return labels

    def _prepare_b2b_dataset_labels(self, labels):
        if labels is None:
            raise RuntimeError(
                "--alg_b2b_multi_dataset_class_conditioning requires "
                "multi_dataset samples with dataset_index"
            )

        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        labels = self._prepare_b2b_labels(labels)
        if torch.any(labels < 0) or torch.any(labels >= self.num_classes):
            raise RuntimeError(
                "multi_dataset dataset_index labels must be in "
                f"[0, G_vit_num_classes - 1], got min={labels.min().item()} "
                f"max={labels.max().item()} with G_vit_num_classes={self.num_classes}"
            )
        return labels

    def _apply_b2b_diff_augment(self):
        if not self.opt.isTrain or not hasattr(self, "diff_augment"):
            return

        aug_images, aug_masks = self.diff_augment.apply_synchronized(
            image_tensors=[self.gt_image, self.y_t],
            mask_tensors=[self.mask] if self.mask is not None else [],
        )
        self.gt_image = aug_images[0]
        self.y_t = aug_images[1]
        if self.mask is not None:
            self.mask = aug_masks[0]

    def _degrade_b2b_reference_frames(self, y_t, use_gt, ref_idx):
        prob = getattr(self.opt, "alg_b2b_ref_degrade_prob", 0.0)
        std = getattr(self.opt, "alg_b2b_ref_degrade_noise_std", 0.05)
        if prob <= 0.0 or std <= 0.0 or use_gt is None or ref_idx is None:
            return y_t

        degrade = use_gt & (torch.rand(use_gt.shape, device=use_gt.device) < prob)
        if not degrade.any():
            return y_t

        y_t = y_t.clone()
        batch_idx = torch.arange(y_t.shape[0], device=y_t.device)
        sel = batch_idx[degrade]
        frame_idx = ref_idx[degrade]
        noisy_ref = y_t[sel, frame_idx] + torch.randn_like(y_t[sel, frame_idx]) * std
        y_t[sel, frame_idx] = noisy_ref.clamp(-1.0, 1.0)
        return y_t

    def _b2b_reference_copy_loss(self, raw_x_pred, target, use_gt, ref_idx):
        if raw_x_pred is None or use_gt is None or ref_idx is None or target.ndim != 5:
            return torch.zeros(size=(), device=target.device)

        use_gt = use_gt.to(device=target.device, dtype=torch.bool).reshape(-1)
        ref_idx = ref_idx.to(device=target.device, dtype=torch.long).reshape(-1)
        if not use_gt.any():
            return torch.zeros(size=(), device=target.device)
        batch_idx = torch.arange(target.shape[0], device=target.device)
        sel = batch_idx[use_gt]
        if sel.numel() == 0:
            return torch.zeros(size=(), device=target.device)

        pred_ref = raw_x_pred[sel, ref_idx[use_gt]]
        target_ref = target[sel, ref_idx[use_gt]]
        if self.opt.alg_b2b_loss in ["MSE", "multiscale_MSE"]:
            return F.mse_loss(pred_ref, target_ref)
        if self.opt.alg_b2b_loss in ["L1", "multiscale_L1"]:
            return F.l1_loss(pred_ref, target_ref)
        if self.opt.alg_b2b_loss == "pseudo_huber":
            return self._pseudo_huber_loss(pred_ref, target_ref)
        raise NotImplementedError(
            f"Unsupported alg_b2b_loss for reference copy loss: {self.opt.alg_b2b_loss}"
        )

    def compute_b2b_loss(self):
        y_0 = self.gt_image
        y_cond = self.cond_image
        mask = self.mask
        B = y_0.shape[0]
        use_perceptual = self.opt.alg_b2b_perceptual_loss != [""]
        ref_copy_use_gt = getattr(self, "use_gt", None)
        ref_copy_idx = getattr(self, "ref_idx", None)
        use_ref_copy = (
            getattr(self.opt, "alg_b2b_lambda_ref_copy", 0.0) > 0
            and y_0.ndim == 5
            and ref_copy_use_gt is not None
            and ref_copy_idx is not None
            and bool(ref_copy_use_gt.any().item())
        )

        if self.num_classes > 0:
            labels = self.label_cls
        else:
            labels = None

        net_kwargs = {}
        if getattr(self.opt, "alg_b2b_temporal_frame_step_conditioning", False):
            net_kwargs["temporal_frame_step"] = getattr(
                self, "temporal_frame_step", None
            )
        if b2b_global_context_enabled(b2b_global_context_mode_from_opt(self.opt)):
            net_kwargs["global_context"] = getattr(self, "global_context", None)
        if getattr(self, "object_refs", None) is not None:
            net_kwargs["object_refs"] = self.object_refs

        net_out = self.netG_A(
            y_0,
            mask,
            y_cond,
            label=labels,
            use_gt=getattr(self, "use_gt", None),
            ref_idx=getattr(self, "ref_idx", None),
            return_x_pred=use_perceptual or use_ref_copy,
            return_raw_x_pred=use_ref_copy,
            **net_kwargs,
        )
        raw_x_pred = None
        if use_ref_copy:
            v_pred, v, x_pred, raw_x_pred = net_out
        elif use_perceptual:
            v_pred, v, x_pred = net_out
        else:
            v_pred, v = net_out
            x_pred = None

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
        self.loss_G_ref_copy = torch.zeros(size=(), device=y_0.device)

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

        if use_ref_copy:
            ref_copy_loss = self._b2b_reference_copy_loss(
                raw_x_pred,
                y_0,
                ref_copy_use_gt,
                ref_copy_idx,
            )
            self.loss_G_ref_copy = self.opt.alg_b2b_lambda_ref_copy * ref_copy_loss
            self.loss_G_tot += self.loss_G_ref_copy

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
        global_context = getattr(self, "global_context", None)
        if global_context is not None:
            global_context = global_context[:nb_imgs]
        temporal_frame_step = getattr(self, "temporal_frame_step", None)
        if temporal_frame_step is not None:
            temporal_frame_step = temporal_frame_step[:nb_imgs]
        object_refs = getattr(self, "object_refs", None)
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
                        y_t,
                        y_cond,
                        self.opt.alg_b2b_denoise_timesteps,
                        cur_mask,
                        temporal_frame_step=temporal_frame_step,
                        global_context=global_context,
                        object_refs=object_refs,
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
                        temporal_frame_step=temporal_frame_step,
                        global_context=global_context,
                        object_refs=object_refs,
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
