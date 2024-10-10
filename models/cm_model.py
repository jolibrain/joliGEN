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

from util.mask_generation import random_edge_mask

from . import diffusion_networks
from .base_diffusion_model import BaseDiffusionModel

from piq import DISTS, LPIPS


def pseudo_huber_loss(input, target):
    """Computes the pseudo huber loss.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    target : Tensor
        Target tensor.

    Returns
    -------
    Tensor
        Pseudo huber loss.
    """
    c = 0.00054 * math.sqrt(math.prod(input.shape[1:]))
    return torch.sqrt((input - target) ** 2 + c**2) - c


class CMModel(BaseDiffusionModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific to the consistency model"""
        parser = BaseDiffusionModel.modify_commandline_options(
            parser, is_train=is_train
        )

        parser.add_argument(
            "--alg_cm_num_steps",
            type=int,
            default=1000000,
            help="number of steps before reaching the fully discretized consistency model sampling schedule",
        )
        parser.add_argument(
            "--alg_cm_perceptual_loss",
            type=str,
            default=[""],
            nargs="*",
            choices=["", "LPIPS", "DISTS"],
            help="optional supervised perceptual loss",
        )
        parser.add_argument(
            "--alg_cm_lambda_perceptual",
            type=float,
            default=1.0,
            help="weight for LPIPS and DISTS perceptual losses",
        )
        parser.add_argument(
            "--alg_cm_dists_mean",
            default=[0.485, 0.456, 0.406],  # Imagenet default
            nargs="*",
            type=float,
            help="mean for DISTS perceptual loss",
        )
        parser.add_argument(
            "--alg_cm_dists_std",
            default=[0.229, 0.224, 0.225],  # Imagenet default
            nargs="*",
            type=float,
            help="std for DISTS perceptual loss",
        )

        if is_train:
            parser = CMModel.modify_commandline_options_train(parser)

        return parser

    @staticmethod
    def modify_commandline_options_train(parser):
        parser = BaseDiffusionModel.modify_commandline_options_train(parser)
        return parser

    def after_parse(opt):
        return opt

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        self.task = self.opt.alg_diffusion_task

        if opt.isTrain:
            batch_size = self.opt.train_batch_size
        else:
            batch_size = self.opt.test_batch_size

        self.total_t = (
            self.opt.alg_cm_num_steps * self.opt.train_batch_size
        )  # scaled with bs

        # Visuals
        visual_outputs = []
        self.gen_visual_names = [
            "gt_image_",
            "y_t_",
            "output_",
            "next_noisy_x_",
            "current_noisy_x_",
        ]

        if opt.alg_diffusion_cond_image_creation != "y_t":
            self.gen_visual_names.append("cond_image_")
        if self.opt.alg_diffusion_cond_image_creation == "previous_frame":
            self.gen_visual_names.insert(0, "previous_frame_")

        for k in range(self.opt.train_batch_size):
            self.visual_names.append([temp + str(k) for temp in self.gen_visual_names])
        self.visual_names.append(visual_outputs)

        # Define network
        opt.alg_palette_sampling_method = ""
        opt.alg_diffusion_cond_embed = opt.alg_diffusion_cond_image_creation
        opt.alg_diffusion_cond_embed_dim = 256
        self.netG_A = diffusion_networks.define_G(**vars(opt)).to(self.device)
        if opt.isTrain:
            self.netG_A.current_t = max(self.netG_A.current_t, opt.total_iters)
        else:
            self.netG_A.current_t = 0  # placeholder
        print("Setting CM current_iter to", self.netG_A.current_t)

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

        self.loss_names_G = ["G_tot"]
        self.loss_names = self.loss_names_G

        # Make group
        self.networks_groups = []

        losses_backward = ["loss_G_tot"]

        self.group_G = NetworkGroup(
            networks_to_optimize=G_models,
            forward_functions=[],
            backward_functions=["compute_cm_loss"],
            loss_names_list=["loss_names_G"],
            optimizer=["optimizer_G"],
            loss_backward=losses_backward,
            networks_to_ema=G_models,
        )
        self.networks_groups.append(self.group_G)

        losses_G = []
        if opt.alg_cm_perceptual_loss != [""]:
            losses_G += ["G_perceptual"]
        self.loss_names_G += losses_G
        self.loss_names = self.loss_names_G.copy()

        # Itercalculator
        self.iter_calculator_init()

        # perceptual losses
        if "LPIPS" in self.opt.alg_cm_perceptual_loss:
            self.criterionLPIPS = LPIPS().to(self.device)
        if "DISTS" in self.opt.alg_cm_perceptual_loss:
            self.criterionDISTS = DISTS(
                mean=self.opt.alg_cm_dists_mean, std=self.opt.alg_cm_dists_std
            ).to(self.device)

    def set_input(self, data):
        if (
            len(data["A"].to(self.device).shape) == 5
        ):  # we're using temporal successive frames
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

        if self.opt.alg_diffusion_cond_image_creation == "previous_frame":
            cond_image_list = []
            for cur_frame, cur_mask in zip(
                self.previous_frame.cpu(),
                self.previous_frame_mask.cpu(),
            ):
                if (
                    random.random()
                    < self.opt.alg_diffusion_cond_prob_use_previous_frame
                ):
                    cond_image_list.append(cur_frame.to(self.device))
                else:
                    cond_image_list.append(
                        -1 * torch.ones_like(cur_frame, device=self.device)
                    )

                self.cond_image = torch.stack(cond_image_list)
                self.cond_image = self.cond_image.to(self.device)
        elif self.opt.alg_diffusion_cond_image_creation == "computed_sketch":
            fill_img_with_random_sketch = random_edge_mask(
                fn_list=self.opt.alg_diffusion_cond_computed_sketch_list
            )
            if "canny" in fill_img_with_random_sketch.__name__:
                low = min(self.opt.alg_diffusion_cond_sketch_canny_range)
                high = max(self.opt.alg_diffusion_cond_sketch_canny_range)
                self.cond_image = fill_img_with_random_sketch(
                    self.gt_image,
                    self.mask,
                    low_threshold_random=low,
                    high_threshold_random=high,
                )
        elif self.task == "pix2pix":
            self.cond_image = self.y_t
        else:  # y_t
            self.cond_image = None

        self.batch_size = self.y_t.shape[0]

        self.real_A = self.y_t
        self.real_B = self.gt_image

    def compute_cm_loss(self):
        y_0 = self.gt_image  # ground truth
        y_cond = self.cond_image  # conditioning
        mask = self.mask
        (
            self.pred_x,
            target_x,
            num_timesteps,
            sigmas,
            loss_weights,
            self.next_noisy_x,
            self.current_noisy_x,
        ) = self.netG_A(y_0, self.total_t, mask, y_cond)

        if mask is not None:
            mask_pred_x = mask * self.pred_x
            mask_target_x = mask * target_x
        else:
            mask_pred_x = self.pred_x
            mask_target_x = target_x
        loss = (pseudo_huber_loss(mask_pred_x, mask_target_x) * loss_weights).mean()

        self.loss_G_tot = loss * self.opt.alg_diffusion_lambda_G

        # perceptual losses, if any
        if "LPIPS" in self.opt.alg_cm_perceptual_loss:
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
        if "DISTS" in self.opt.alg_cm_perceptual_loss:
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
            self.loss_G_perceptual = self.opt.alg_cm_lambda_perceptual * (
                self.loss_G_perceptual_lpips + self.loss_G_perceptual_dists
            )
            self.loss_G_tot += self.loss_G_perceptual

    def inference(self, nb_imgs, offset=0):

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
        sampling_sigmas = (80.0, 24.4, 5.84, 0.9, 0.661)
        if not self.cond_image is None:
            y_cond = self.cond_image[:nb_imgs]
        else:
            y_cond = None
        if (
            self.task == "pix2pix"
        ):  # y_t must be of output channel size, since we do not have y_0 (gt), we get it from the model
            out_shape = list(y_cond.shape)
            out_shape[1] = netG.cm_model.out_channel
            y_t = torch.zeros(out_shape, device=y_cond.device, dtype=y_cond.dtype)
        else:  # e.g. inpainting
            y_t = self.y_t[:nb_imgs]
        self.output = netG.restoration(y_t, y_cond, sampling_sigmas, mask)
        self.fake_B = self.output
        self.visuals = self.output

        # set visual names
        if self.opt.isTrain:
            for name in self.gen_visual_names:
                whole_tensor = getattr(self, name[:-1])
                for k in range(min(nb_imgs, self.get_current_batch_size())):
                    cur_name = name + str(offset + k)
                    cur_tensor = whole_tensor[k : k + 1]
                    if "mask" in name:
                        cur_tensor = cur_tensor.squeeze(0)
                    setattr(self, cur_name, cur_tensor)

    def compute_visuals(self, nb_imgs):
        super().compute_visuals(nb_imgs)
        with torch.no_grad():
            self.inference(nb_imgs)
