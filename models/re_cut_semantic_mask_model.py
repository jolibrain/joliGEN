import torch
import itertools
from .cut_semantic_mask_model import CUTSemanticMaskModel
from . import gan_networks
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .modules import loss
from util.iter_calculator import IterCalculator
from util.network_group import NetworkGroup


class ReCUTSemanticMaskModel(CUTSemanticMaskModel):
    def modify_commandline_options(parser, is_train=True):
        parser = CUTSemanticMaskModel.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        if self.opt.alg_re_adversarial_loss_p:
            self.loss_names_G += ["proj_fake_B_adversarial"]
        self.loss_names_G += ["recut"]
        self.loss_names_P = ["proj_real_B"]
        if self.opt.alg_re_adversarial_loss_p:
            self.loss_names_P += ["proj_real_A_adversarial", "proj_real_B_adversarial"]

        self.loss_names = (
            self.loss_names_G
            + self.loss_names_f_s
            + self.loss_names_D
            + self.loss_names_P
        )

        if self.opt.train_iter_size > 1:
            self.iter_calculator = IterCalculator(self.loss_names)
            for i, cur_loss in enumerate(self.loss_names):
                self.loss_names[i] = cur_loss + "_avg"
                setattr(self, "loss_" + self.loss_names[i], 0)

        self.visual_names += [
            ["real_A_last", "proj_fake_B"],
            ["real_B_last", "proj_real_B"],
        ]

        self.netP_B = networks.define_G(
            (self.opt.alg_re_nuplet_size - 1) * opt.model_input_nc,
            opt.model_output_nc,
            opt.G_ngf,
            opt.alg_re_netP,
            opt.G_norm,
            opt.G_dropout,
            opt.G_spectral,
            opt.model_init_type,
            opt.model_init_gain,
            self.gpu_ids,
            padding_type=opt.G_padding_type,
            opt=self.opt,
        )
        self.model_names += ["P_B"]

        self.optimizer_P = opt.optim(
            opt,
            itertools.chain(self.netP_B.parameters()),
            lr=opt.alg_re_P_lr,
            betas=(opt.train_beta1, opt.train_beta2),
        )
        self.optimizers.append(self.optimizer_P)

        if self.opt.alg_re_no_train_P_fake_images:
            self.group_P = NetworkGroup(
                networks_to_optimize=["P_B"],
                forward_functions=["forward_P"],
                backward_functions=["compute_P_loss"],
                loss_names_list=["loss_names_P"],
                optimizer=["optimizer_P"],
                loss_backward=["loss_P"],
            )
            self.networks_groups.insert(1, self.group_P)
        else:  # P and G networks will be trained in the same time
            self.group_G = NetworkGroup(
                networks_to_optimize=["G", "P_B"],
                forward_functions=["forward", "forward_P"],
                backward_functions=["compute_G_loss", "compute_P_loss"],
                loss_names_list=["loss_names_G", "loss_names_P"],
                optimizer=["optimizer_G", "optimizer_P"],
                loss_backward=["loss_G", "loss_P"],
                networks_to_ema=["G"],
            )
            self.networks_groups[0] = self.group_G

        self.criterionCycle = torch.nn.L1Loss()

    def set_input(self, input):
        super().set_input(input)
        batch_size, nuplet, channels, h, w = self.real_A.shape
        self.shape_fake = [batch_size, (nuplet - 1) * channels, h, w]
        self.real_A_last = self.real_A[:, -1]
        self.real_B_last = self.real_B[:, -1]
        self.input_A_label_last = self.input_A_label[:, -1]
        self.input_B_label_last = self.input_B_label[:, -1]

        self.real_A = self.real_A[:, :-1]
        self.real_B = self.real_B[:, :-1]
        self.input_A_label = self.input_A_label[:, :-1]
        self.input_B_label = self.input_B_label[:, :-1]

        self.real_A = torch.flatten(self.real_A, start_dim=0, end_dim=1)
        self.real_B = torch.flatten(self.real_B, start_dim=0, end_dim=1)
        self.input_A_label = torch.flatten(self.input_A_label, start_dim=0, end_dim=2)
        self.input_B_label = torch.flatten(self.input_B_label, start_dim=0, end_dim=2)

    def data_dependent_initialize(self, data):
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0)
        self.shape_fake[0] = bs_per_gpu
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.input_A_label = self.input_A_label[:bs_per_gpu]
        if hasattr(self, "input_B_label"):
            self.input_B_label = self.input_B_label[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        self.forward_P()
        if self.opt.isTrain:
            self.compute_P_loss()
            self.compute_D_loss()
            self.compute_f_s_loss()
            self.compute_G_loss()
            self.loss_D.backward()  # calculate gradients for D
            self.loss_f_s.backward()  # calculate gradients for f_s
            self.loss_G.backward()  # calculate gradients for G
            self.loss_P.backward()  # calculate gradients for P
            if self.opt.alg_cut_lambda_NCE > 0.0:
                self.optimizer_F = opt.optim(
                    opt,
                    self.netF.parameters(),
                    lr=self.opt.train_G_lr,
                    betas=(self.opt.train_beta1, self.opt.train_beta2),
                )
                self.optimizers.append(self.optimizer_F)

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        visual_names_seg_A = ["input_A_label", "gt_pred_A", "pfB_max"]

        if hasattr(self, "input_B_label"):
            visual_names_seg_B = ["input_B_label"]
        else:
            visual_names_seg_B = []

        visual_names_seg_B += ["gt_pred_B"]

        self.visual_names += [visual_names_seg_A, visual_names_seg_B]

        if self.opt.train_mask_out_mask and self.isTrain:
            visual_names_out_mask_A = ["real_A_out_mask", "fake_B_out_mask"]
            self.visual_names += [visual_names_out_mask_A]

    def forward(self):
        super().forward()
        ## Projection next time step over fake images
        self.proj_fake_B = self.netP_B(self.fake_B.reshape(self.shape_fake))

    def compute_G_loss(self):
        super().compute_G_loss()

        ## GAN loss over fake images projection for G (and P if no_train_P_fake_images is False)
        if self.opt.alg_re_adversarial_loss_p:
            self.loss_proj_fake_B_adversarial = self.compute_G_loss_GAN_generic(
                self.netD, "B", self.D_loss, fake_name="proj_fake_B"
            )
        else:
            self.loss_proj_fake_B_adversarial = 0

        ## Recycle loss between fake images projection reconstruction and ground truth
        if (
            not hasattr(self, "loss_proj_real_B")
            or (self.loss_proj_real_B) > self.opt.alg_re_projection_threshold
        ):  # if P networks aren't accurate enough on real images, we don't use them on fake images:
            self.loss_recut = 0
            self.loss_proj_fake_B_adversarial = 0

        else:
            self.loss_recut = self.calculate_NCE_loss(
                self.real_A_last, self.proj_fake_B
            )

        self.loss_G += self.loss_proj_fake_B_adversarial + self.loss_recut

    def forward_P(self):
        ## Real images projection
        self.proj_real_B = self.netP_B(self.real_B.reshape(self.shape_fake))

    def compute_P_loss(self):
        ## Pixel to pixel loss between real images projection and ground truth
        lambda_P = 10
        self.loss_proj_real_B = (
            self.criterionCycle(self.proj_real_B, self.real_B_last) * lambda_P
        )

        ## GAN loss over real images projection for P
        if self.opt.alg_re_adversarial_loss_p:
            self.loss_proj_real_B_adversarial = self.compute_G_loss_GAN_generic(
                self.netD, "B", self.D_loss, fake_name="proj_real_B"
            )
        else:
            self.loss_proj_real_B_adversarial = 0

        self.loss_P = self.loss_proj_real_B + self.loss_proj_real_B_adversarial

    def compute_D_loss(self):
        super().compute_D_loss()
        ## GAN loss over fake images projections for D
        self.loss_D += self.compute_D_loss_generic(
            self.netD,
            "B",
            self.D_loss,
            real_name="real_B_last",
            fake_name="proj_fake_B",
        )

        ## GAN loss over real images projections for D
        self.loss_D += self.compute_D_loss_generic(
            self.netD,
            "B",
            self.D_loss,
            real_name="real_B_last",
            fake_name="proj_real_B",
        )

        if self.opt.D_netD_global != "none":
            ## GAN loss over fake images projections for D_global
            self.loss_D_global += self.compute_D_loss_generic(
                self.netD_global,
                "B",
                self.D_global_loss,
                real_name="real_B_last",
                fake_name="proj_fake_B",
            )

            ## GAN loss over real images projections for D_global
            self.loss_D_global += self.compute_D_loss_generic(
                self.netD_global,
                "B",
                self.D_global_loss,
                real_name="real_B_last",
                fake_name="proj_real_B",
            )

        self.loss_D = self.loss_D + self.loss_D_global
