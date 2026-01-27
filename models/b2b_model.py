import copy
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
            choices=[1, 2, 4, 8, 16, 32, 50, 64, 128],
            help="Number of denoising steps at inference",
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
            choices=["L1", "MSE", "multiscale_L1", "multiscale_MSE"],
            help="Loss type for B2B denoising",
        )

        return parser

    @staticmethod
    def after_parse(opt):
        # Example: validate incompatible options here
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

                if use_gt.any():
                    batch_idx = torch.arange(B, device=self.device)
                    sel = batch_idx[use_gt]  # selected samples only

                    # one random frame index per sample (we'll only use those for sel)
                    idx = torch.randint(0, T, (B,), device=self.device)

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

        # base "real" labels in [0 .. num_classes-1]
        if self.num_classes > 0:
            labels = torch.zeros(
                B, dtype=torch.long, device=self.device
            )  # class 1 start from 0, need discuss
        else:
            labels = None

        v_pred, v = self.netG_A(y_0, mask, y_cond, label=labels)

        if not self.opt.alg_b2b_minsnr:
            min_snr_loss_weight = 1.0

        if mask is not None:
            mask_binary = torch.clamp(mask, min=0, max=1)
            loss = self.loss_fn(
                min_snr_loss_weight * mask_binary * v_pred,
                min_snr_loss_weight * mask_binary * v,
            )
        else:
            loss = self.loss_fn(min_snr_loss_weight * v_pred, min_snr_loss_weight * v)

        if isinstance(loss, dict):
            loss_tot = torch.zeros(size=(), device=noise.device)

            for cur_size, cur_loss in loss.items():
                setattr(self, "loss_G_" + cur_size, cur_loss)
                loss_tot += cur_loss

            loss = loss_tot

        self.loss_G_tot = self.opt.alg_diffusion_lambda_G * loss

        # perceptual losses, if any
        if "LPIPS" in self.opt.alg_b2b_perceptual_loss:
            if mask_pred_x.size(1) > 3:  # more than 3 channels
                self.loss_G_perceptual_lpips = 0.0
                for c in range(4):  # per channel loss and sum
                    y_0_Bc = y_0[:, c, :, :].unsqueeze(1)
                    mask_pred_Bc = mask_pred_x[:, c, :, :].unsqueeze(1)
                    self.loss_G_perceptual_lpips += self.criterionLPIPS(
                        y_0_Bc, mask_pred_Bc
                    )
            else:
                self.loss_G_perceptual_lpips = torch.mean(
                    self.criterionLPIPS(y_0, mask_pred_x)
                )
        else:
            self.loss_G_perceptual_lpips = 0
        if "DISTS" in self.opt.alg_b2b_perceptual_loss:
            if mask_pred_x.size(1) > 3:  # more than 3 channels
                self.loss_G_perceptual_dists = 0.0
                for c in range(4):  # per channel loss and sum
                    y_0_Bc = y_0[:, c, :, :].unsqueeze(1)
                    mask_pred_Bc = mask_pred_x[:, c, :, :].unsqueeze(1)
                    self.loss_G_perceptual_dists += self.criterionDISTS(
                        y_0_Bc, mask_pred_Bc
                    )
            else:
                self.loss_G_perceptual_dists = self.criterionDISTS(y_0, mask_pred_x)
        else:
            self.loss_G_perceptual_dists = 0

        if self.loss_G_perceptual_lpips > 0 or self.loss_G_perceptual_dists > 0:
            self.loss_G_perceptual = self.opt.alg_b2b_lambda_perceptual * (
                self.loss_G_perceptual_lpips + self.loss_G_perceptual_dists
            )
            self.loss_G_tot += self.loss_G_perceptual

    def inference(self, nb_imgs, offset=0):
        offset = 0
        if hasattr(self.netG_A, "module"):
            netG = self.netG_A.module
        else:
            netG = self.netG_A

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

                for steps in self.opt.alg_b2b_denoise_timesteps:
                    self.output = netG.restoration(
                        y_t, y_cond, steps, mask, self.label_cls
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
