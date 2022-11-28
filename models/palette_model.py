import torch
import tqdm

from .base_diffusion_model import BaseDiffusionModel
from util.network_group import NetworkGroup
from util.iter_calculator import IterCalculator
from . import diffusion_networks

import copy

import warnings


class PaletteModel(BaseDiffusionModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific to the Palette model"""
        parser = BaseDiffusionModel.modify_commandline_options(
            parser, is_train=is_train
        )
        parser.add_argument(
            "--alg_palette_lambda_G",
            type=float,
            default=1.0,
            help="weight for supervised loss",
        )
        parser.add_argument(
            "--alg_palette_loss",
            type=str,
            default="MSE",
            choices=["L1", "MSE"],
            help="loss for denoising model",
        )
        return parser

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        # Visuals
        self.visual_names.append(["gt_image", "cond_image", "mask", "output"])

        if opt.G_nblocks == 9:
            warnings.warn(
                f"G_nblocks default value {opt.G_nblocks} is too high for palette model, 2 will be used instead."
            )
            opt.G_nblocks = 2

        # Define networks
        self.netG_A = diffusion_networks.define_G(**vars(opt))

        self.model_names = ["G_A"]

        self.model_names_export = ["G_A"]

        # Define optimizer
        self.optimizer_G = opt.optim(
            opt,
            self.netG_A.parameters(),
            lr=opt.train_G_lr,
            betas=(opt.train_beta1, opt.train_beta2),
        )

        self.optimizers.append(self.optimizer_G)

        # Define loss functions
        if self.opt.alg_palette_loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        elif self.opt.alg_palette_loss == "L1":
            self.loss_fn = torch.nn.L1Loss()

        losses_G = ["G_tot"]

        self.loss_names_G = losses_G
        self.loss_names = self.loss_names_G

        # Make group
        self.networks_groups = []

        losses_backward = ["loss_G_tot"]

        self.group_G = NetworkGroup(
            networks_to_optimize=["G_A"],
            forward_functions=[],
            backward_functions=["compute_palette_loss"],
            loss_names_list=["loss_names_G"],
            optimizer=["optimizer_G"],
            loss_backward=losses_backward,
            networks_to_ema=["G_A"],
        )
        self.networks_groups.append(self.group_G)

        losses_G = []

        self.loss_names_G += losses_G

        self.loss_names = self.loss_names_G.copy()

        # Itercalculator
        if self.opt.train_iter_size > 1:

            self.iter_calculator = IterCalculator(self.loss_names)
            for i, cur_loss in enumerate(self.loss_names):
                self.loss_names[i] = cur_loss + "_avg"
                setattr(self, "loss_" + self.loss_names[i], 0)

        self.sample_num = 2

    def set_input(self, data):
        """must use set_device in tensor"""

        self.cond_image = data["A"].to(self.device)
        self.gt_image = data["B"].to(self.device)
        self.mask = data["B_label_mask"].to(self.device)
        self.batch_size = self.cond_image.shape[0]

        self.real_A = self.cond_image
        self.real_B = self.gt_image

    def compute_palette_loss(self):
        y_0 = self.gt_image
        y_cond = self.cond_image
        mask = self.mask
        noise = None

        noise, noise_hat = self.netG_A(y_0, y_cond, mask, noise)

        if mask is not None:
            loss = self.loss_fn(mask * noise, mask * noise_hat)
        else:
            loss = self.loss_fn(noise, noise_hat)

        self.loss_G_tot = self.opt.alg_palette_lambda_G * loss

    def inference(self):
        if hasattr(self.netG_A, "module"):
            netG = self.netG_A.module
        else:
            netG = self.netG_A
        if True or self.task in ["inpainting", "uncropping"]:
            self.output, self.visuals = netG.restoration(
                self.cond_image,
                y_t=self.cond_image,
                y_0=self.gt_image,
                mask=self.mask,
                sample_num=self.sample_num,
            )
        else:
            self.output, self.visuals = self.restoration(
                self.cond_image, sample_num=self.sample_num
            )

        self.fake_B = self.visuals[-1:]

    def compute_visuals(self):
        super().compute_visuals()
        with torch.no_grad():
            self.inference()
