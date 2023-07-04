import numpy as np
import torch
import torch.nn.functional as F

from .base_gan_model import BaseGanModel
from . import gan_networks

from .modules import loss


from util.network_group import NetworkGroup
import util.util as util
from util.util import gaussian

import itertools


class CycleGanModel(BaseGanModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific for cyclegan model"""
        parser = BaseGanModel.modify_commandline_options(parser, is_train=is_train)
        parser.set_defaults(G_dropout=False)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument(
                "--alg_cyclegan_lambda_A",
                type=float,
                default=10.0,
                help="weight for cycle loss (A -> B -> A)",
            )
            parser.add_argument(
                "--alg_cyclegan_lambda_B",
                type=float,
                default=10.0,
                help="weight for cycle loss (B -> A -> B)",
            )
            parser.add_argument(
                "--alg_cyclegan_lambda_identity",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
            )
            parser.add_argument(
                "--alg_cyclegan_rec_noise",
                type=float,
                default=0.0,
                help="whether to add noise to reconstruction",
            )
        return parser

    @staticmethod
    def after_parse(opt):
        return opt

    def __init__(self, opt, rank):

        super().__init__(opt, rank)

        if opt.alg_cyclegan_lambda_identity > 0.0:
            # only works when input and output images have the same number of channels
            assert opt.model_input_nc == opt.model_output_nc

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if (
            self.isTrain and self.opt.alg_cyclegan_lambda_identity > 0.0
        ):  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = [
            visual_names_A,
            visual_names_B,
        ] + self.visual_names  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.

        if self.opt.dataaug_diff_aug_policy != "":
            self.visual_names.append(["real_A_aug", "fake_B_aug"])
            self.visual_names.append(["real_B_aug", "fake_A_aug"])

        if self.opt.output_display_diff_fake_real:
            self.visual_names.append(["diff_real_B_fake_A", "diff_real_A_fake_B"])

        # Models names

        if self.isTrain:
            self.model_names = ["G_A", "G_B"]
            self.model_names_export = ["G_A"]
        else:  # during test time, only load Gs
            self.model_names = ["G_A", "G_B"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        # Generators

        self.netG_A = gan_networks.define_G(**vars(opt))
        self.netG_B = gan_networks.define_G(**vars(opt))

        # Discriminators

        if self.isTrain:

            self.netD_As = gan_networks.define_D(**vars(opt))
            self.netD_Bs = gan_networks.define_D(**vars(opt))

            discriminators_names_A = ["D_A_" + D_name for D_name in self.netD_As.keys()]
            discriminators_names_B = ["D_B_" + D_name for D_name in self.netD_Bs.keys()]
            self.discriminators_names = discriminators_names_A + discriminators_names_B
            self.model_names += discriminators_names_A
            self.model_names += discriminators_names_B

            for D_name, netD in self.netD_As.items():
                setattr(self, "netD_A_" + D_name, netD)

            for D_name, netD in self.netD_Bs.items():
                setattr(self, "netD_B_" + D_name, netD)

            # Define loss functions
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # Optimizers
            self.optimizer_G = opt.optim(
                opt,
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.train_G_lr,
                betas=(opt.train_beta1, opt.train_beta2),
                weight_decay=opt.train_optim_weight_decay,
                eps=opt.train_optim_eps,
            )

            D_parameters = itertools.chain(
                *[
                    getattr(self, "net" + D_name).parameters()
                    for D_name in self.discriminators_names
                ]
            )

            self.optimizer_D = opt.optim(
                opt,
                D_parameters,
                lr=opt.train_D_lr,
                betas=(opt.train_beta1, opt.train_beta2),
                weight_decay=opt.train_optim_weight_decay,
                eps=opt.train_optim_eps,
            )

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Making groups

            self.networks_groups = []

            self.group_G = NetworkGroup(
                networks_to_optimize=["G_A", "G_B"],
                forward_functions=["forward"],
                backward_functions=["compute_G_loss"],
                loss_names_list=["loss_names_G"],
                optimizer=["optimizer_G"],
                loss_backward=["loss_G_tot"],
                networks_to_ema=["G_A", "G_B"],
            )
            self.networks_groups.append(self.group_G)

            self.group_D = NetworkGroup(
                networks_to_optimize=self.discriminators_names,
                forward_functions=None,
                backward_functions=["compute_D_loss"],
                loss_names_list=["loss_names_D"],
                optimizer=["optimizer_D"],
                loss_backward=["loss_D_tot"],
            )
            self.networks_groups.append(self.group_D)

            # Discriminators

            self.set_discriminators_info()

            # Losses names

            losses_G = ["G_cycle_A", "G_idt_A", "G_cycle_B", "G_idt_B"]
            losses_D = []

            for discriminator in self.discriminators:
                losses_G.append(discriminator.loss_name_G)
                losses_D.append(discriminator.loss_name_D)

            self.loss_names_G += losses_G
            self.loss_names_D += losses_D

            self.loss_names = self.loss_names_G + self.loss_names_D

        if self.opt.data_online_context_pixels > 0:
            self.context_visual_names_A = [
                "real_A_with_context_vis",
                "fake_B_with_context_vis",
                "mask_context_vis",
            ]

            self.context_visual_names_B = [
                "real_B_with_context_vis",
                "fake_A_with_context_vis",
                "mask_context_vis",
            ]

            self.visual_names.append(self.context_visual_names_A)
            self.visual_names.append(self.context_visual_names_B)

        if self.opt.train_semantic_mask:
            self.init_semantic_mask(opt)
        if self.opt.train_semantic_cls:
            self.init_semantic_cls(opt)

        self.loss_functions_G.append("compute_G_loss_cycle_gan")
        self.forward_functions.insert(1, "forward_cycle_gan")

        # Itercalculator
        self.iter_calculator_init()

    def data_dependent_initialize(self, data):
        self.set_input_first_gpu(data)
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        if self.opt.train_semantic_mask:
            self.data_dependent_initialize_semantic_mask(data)

    def data_dependent_initialize_semantic_mask(self, data):
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_seg_A = ["input_A_label_mask", "gt_pred_f_s_real_A_max", "pfB_max"]

        if hasattr(self, "input_B_label_mask"):
            visual_names_seg_B = ["input_B_label_mask"]
        else:
            visual_names_seg_B = []

        visual_names_seg_B += ["gt_pred_f_s_real_B_max", "pfA_max"]

        self.visual_names += [visual_names_seg_A, visual_names_seg_B]

        if self.opt.train_mask_out_mask and self.isTrain:
            visual_names_out_mask_A = ["real_A_out_mask", "fake_B_out_mask"]
            visual_names_out_mask_B = ["real_B_out_mask", "fake_A_out_mask"]
            self.visual_names += [visual_names_out_mask_A, visual_names_out_mask_B]

    def inference(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)

    def forward_cycle_gan(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        ### Fake B

        self.fake_B = self.netG_A(self.real_A)  # G_A(A)

        if self.opt.data_online_context_pixels > 0:
            if self.use_temporal:
                self.compute_temporal_fake_with_context(
                    fake_name="temporal_fake_B_0", real_name="temporal_real_A_0"
                )
            self.compute_fake_with_context(fake_name="fake_B", real_name="real_A")

        # Rec A

        if self.opt.alg_cyclegan_rec_noise > 0.0:
            self.fake_B_noisy1 = gaussian(self.fake_B, self.opt.alg_cyclegan_rec_noise)
            self.rec_A = self.netG_B(self.fake_B_noisy1)
        else:
            self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))

        # Fake A

        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        if self.opt.data_online_context_pixels > 0:
            self.compute_fake_with_context(fake_name="fake_A", real_name="real_B")

        # Rec B

        if self.opt.alg_cyclegan_rec_noise > 0.0:
            self.fake_A_noisy1 = gaussian(self.fake_A, self.opt.alg_cyclegan_rec_noise)
            self.rec_B = self.netG_A(self.fake_A_noisy1)
        else:
            self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

        if self.opt.dataaug_D_noise > 0.0:
            context = ""
            if self.opt.data_online_context_pixels > 0:
                context = "_with_context"

            names = ["fake_B", "real_A", "fake_A", "real_B"]
            for name in names:
                setattr(
                    self,
                    name + context + "_noisy",
                    gaussian(getattr(self, name + context), self.opt.dataaug_D_noise),
                )

        if self.opt.alg_cyclegan_lambda_identity > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)

        self.diff_real_B_fake_A = self.real_B - self.fake_A
        self.diff_real_A_fake_B = self.real_A - self.fake_B

    def compute_G_loss_cycle_gan(self):
        """Calculate the loss for generators G_A and G_B"""

        lambda_idt = self.opt.alg_cyclegan_lambda_identity
        lambda_A = self.opt.alg_cyclegan_lambda_A
        lambda_B = self.opt.alg_cyclegan_lambda_B

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.loss_G_idt_A = (
                self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            )
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.loss_G_idt_B = (
                self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            )

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_G_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_G_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G_tot += (
            +self.loss_G_cycle_A
            + self.loss_G_cycle_B
            + self.loss_G_idt_A
            + self.loss_G_idt_B
        )
