import torch
import itertools
from util.image_pool import ImagePool
from util.losses import L1_Charbonnier_loss
from .cycle_gan_model import CycleGANModel
from . import networks
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .modules import loss
from util.iter_calculator import IterCalculator
from util.network_group import NetworkGroup


class CycleGANSemanticMaskModel(CycleGANModel):
    def name(self):
        return "CycleGANSemanticMaskModel"

    # new, copied from cyclegansemantic model
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CycleGANModel.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        losses_G = ["sem_AB", "sem_BA"]

        if opt.train_mask_out_mask:
            losses_G += ["out_mask_AB", "out_mask_BA"]

        losses_f_s = ["f_s"]

        losses_D = []

        self.loss_names_G += losses_G
        self.loss_names_f_s = losses_f_s
        self.loss_names_D += losses_D

        self.loss_names = self.loss_names_G + self.loss_names_f_s + self.loss_names_D

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            if self.opt.train_mask_disjoint_f_s:
                self.opt.train_mask_f_s_B = True
                self.model_names += ["f_s_A", "f_s_B"]
            else:
                self.model_names += ["f_s"]

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        if self.isTrain:
            networks_f_s = []
            if self.opt.train_mask_disjoint_f_s:
                self.netf_s_A = networks.define_f(**vars(opt))
                networks_f_s.append("f_s_A")
                self.netf_s_B = networks.define_f(**vars(opt))
                networks_f_s.append("f_s_B")

            else:
                self.netf_s = networks.define_f(**vars(opt))
                networks_f_s.append("f_s")

            self.fake_A_pool_mask = ImagePool(opt.train_pool_size)
            self.fake_B_pool_mask = ImagePool(opt.train_pool_size)

            # define loss functions
            self.criterionf_s = torch.nn.modules.CrossEntropyLoss()

            if opt.train_mask_out_mask:
                if opt.train_mask_loss_out_mask == "L1":
                    self.criterionMask = torch.nn.L1Loss()
                elif opt.train_mask_loss_out_mask == "MSE":
                    self.criterionMask = torch.nn.MSELoss()
                elif opt.train_mask_loss_out_mask == "Charbonnier":
                    self.criterionMask = L1_Charbonnier_loss(
                        opt.train_mask_charbonnier_eps
                    )

            # initialize optimizers
            if self.opt.train_mask_disjoint_f_s:
                self.optimizer_f_s = opt.optim(
                    opt,
                    itertools.chain(
                        self.netf_s_A.parameters(), self.netf_s_B.parameters()
                    ),
                    lr=opt.train_sem_lr_f_s,
                    betas=(opt.train_beta1, opt.train_beta2),
                )
            else:
                self.optimizer_f_s = opt.optim(
                    opt,
                    self.netf_s.parameters(),
                    lr=opt.train_sem_lr_f_s,
                    betas=(opt.train_beta1, opt.train_beta2),
                )

            self.optimizers.append(self.optimizer_f_s)

            if self.opt.train_iter_size > 1:
                self.iter_calculator = IterCalculator(self.loss_names)
                for i, cur_loss in enumerate(self.loss_names):
                    self.loss_names[i] = cur_loss + "_avg"
                    setattr(self, "loss_" + self.loss_names[i], 0)

            ###Making groups
            discriminators = ["D_A", "D_B"]

            self.group_f_s = NetworkGroup(
                networks_to_optimize=networks_f_s,
                forward_functions=None,
                backward_functions=["compute_f_s_loss"],
                loss_names_list=["loss_names_f_s"],
                optimizer=["optimizer_f_s"],
                loss_backward=["loss_f_s"],
            )
            self.networks_groups.append(self.group_f_s)

    def set_input(self, input):
        super().set_input(input)
        if "A_label" in input:
            self.input_A_label = input["A_label"].to(self.device).squeeze(1)

            if self.opt.data_online_context_pixels > 0:
                self.input_A_label = self.input_A_label[
                    :,
                    self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                ]

        if "B_label" in input and len(input["B_label"]) > 0:
            self.input_B_label = (
                input["B_label"].to(self.device).squeeze(1)
            )  # beniz: unused

            if self.opt.data_online_context_pixels > 0:
                self.input_B_label = self.input_B_label[
                    :,
                    self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                    self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                ]

    def data_dependent_initialize(self, data):
        self.set_input(data)
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_seg_A = ["input_A_label", "gt_pred_A", "pfB_max"]

        if hasattr(self, "input_B_label"):
            visual_names_seg_B = ["input_B_label"]
        else:
            visual_names_seg_B = []

        visual_names_seg_B += ["gt_pred_B", "pfA_max"]

        visual_names_out_mask_A = ["real_A_out_mask", "fake_B_out_mask"]

        visual_names_out_mask_B = ["real_B_out_mask", "fake_A_out_mask"]

        visual_names_mask = ["fake_B_mask", "fake_A_mask"]

        visual_names_mask_in = [
            "real_B_mask",
            "fake_B_mask",
            "real_A_mask",
            "fake_A_mask",
            "real_B_mask_in",
            "fake_B_mask_in",
            "real_A_mask_in",
            "fake_A_mask_in",
        ]

        self.visual_names += [visual_names_seg_A, visual_names_seg_B]

        if self.opt.train_mask_out_mask:
            self.visual_names += [visual_names_out_mask_A, visual_names_out_mask_B]

    def forward(self):
        super().forward()
        d = 1
        if self.opt.train_mask_disjoint_f_s:
            f_s = self.netf_s_A
        else:
            f_s = self.netf_s

        if self.isTrain:
            self.pred_real_A = f_s(self.real_A)
            self.gt_pred_A = F.log_softmax(self.pred_real_A, dim=d).argmax(dim=d)

            self.pred_fake_A = f_s(self.fake_A)

            if self.opt.train_mask_disjoint_f_s:
                f_s = self.netf_s_B
            else:
                f_s = self.netf_s

            self.pred_real_B = f_s(self.real_B)
            self.gt_pred_B = F.log_softmax(self.pred_real_B, dim=d).argmax(dim=d)

            self.pfA = F.log_softmax(self.pred_fake_A, dim=d)
            self.pfA_max = self.pfA.argmax(dim=d)

            if hasattr(self, "criterionMask"):
                label_A = self.input_A_label
                label_A_in = label_A.unsqueeze(1)
                label_A_inv = (
                    torch.tensor(np.ones(label_A.size())).to(self.device) - label_A > 0
                )
                label_A_inv = label_A_inv.unsqueeze(1)

                self.real_A_out_mask = self.real_A * label_A_inv
                self.fake_B_out_mask = self.fake_B * label_A_inv

                if hasattr(self, "input_B_label") and len(self.input_B_label) > 0:

                    label_B = self.input_B_label
                    label_B_in = label_B.unsqueeze(1)
                    label_B_inv = (
                        torch.tensor(np.ones(label_B.size())).to(self.device) - label_B
                        > 0
                    )
                    label_B_inv = label_B_inv.unsqueeze(1)

                    self.real_B_out_mask = self.real_B * label_B_inv
                    self.fake_A_out_mask = self.fake_A * label_B_inv
        if self.opt.train_mask_disjoint_f_s:
            f_s = self.netf_s_B
        else:
            f_s = self.netf_s

        self.pred_fake_B = f_s(self.fake_B)
        self.pfB = F.log_softmax(self.pred_fake_B, dim=d)
        self.pfB_max = self.pfB.argmax(dim=d)

    def compute_f_s_loss(self):
        self.loss_f_s = 0
        if not self.opt.train_mask_no_train_f_s_A:
            label_A = self.input_A_label
            # forward only real source image through semantic classifier
            if self.opt.train_mask_disjoint_f_s:
                f_s = self.netf_s_A
            else:
                f_s = self.netf_s

            pred_A = f_s(self.real_A)
            self.loss_f_s += self.criterionf_s(pred_A, label_A)  # .squeeze(1))

        if self.opt.train_mask_f_s_B:
            label_B = self.input_B_label
            if self.opt.train_mask_disjoint_f_s:
                f_s = self.netf_s_B
            else:
                f_s = self.netf_s

            pred_B = f_s(self.real_B)
            self.loss_f_s += self.criterionf_s(pred_B, label_B)  # .squeeze(1))

    def compute_D_A_mask_loss(self):
        fake_B_mask = self.fake_B_pool_mask.query(self.fake_B_mask)
        self.loss_D_A_mask = self.compute_D_loss_basic(
            self.netD_A_mask, self.real_B_mask, fake_B_mask
        )

    def compute_D_B_mask_loss(self):
        fake_A_mask = self.fake_A_pool_mask.query(self.fake_A_mask)
        self.loss_D_B_mask = self.compute_D_loss_basic(
            self.netD_B_mask, self.real_A_mask, fake_A_mask
        )

    def compute_D_A_mask_in_loss(self):
        fake_B_mask_in = self.fake_B_pool.query(self.fake_B_mask_in)
        self.loss_D_A = self.compute_D_loss_basic(
            self.netD_A, self.real_B_mask_in, fake_B_mask_in
        )

    def compute_D_B_mask_in_loss(self):
        fake_A_mask_in = self.fake_A_pool.query(self.fake_A_mask)
        self.loss_D_B = self.compute_D_loss_basic(
            self.netD_B, self.real_A_mask_in, fake_A_mask_in
        )

    def compute_D_loss(self):
        super().compute_D_loss()

    def compute_G_loss(self):
        super().compute_G_loss()

        # semantic loss AB
        self.loss_sem_AB = self.opt.train_sem_lambda * self.criterionf_s(
            self.pfB, self.input_A_label
        )

        # semantic loss BA
        if hasattr(self, "input_B_label"):
            self.loss_sem_BA = self.opt.train_sem_lambda * self.criterionf_s(
                self.pfA, self.input_B_label
            )  # .squeeze(1))
        else:
            self.loss_sem_BA = self.opt.train_sem_lambda * self.criterionf_s(
                self.pfA, self.gt_pred_B
            )  # .squeeze(1))

        # only use semantic loss when classifier has reasonably low loss
        if (
            not hasattr(self, "loss_f_s")
            or self.loss_f_s > self.opt.f_s_semantic_threshold
        ):
            self.loss_sem_AB = 0 * self.loss_sem_AB
            self.loss_sem_BA = 0 * self.loss_sem_BA
        self.loss_G += self.loss_sem_BA + self.loss_sem_AB

        lambda_out_mask = self.opt.train_mask_lambda_out_mask

        if hasattr(self, "criterionMask"):
            self.loss_out_mask_AB = (
                self.criterionMask(self.real_A_out_mask, self.fake_B_out_mask)
                * lambda_out_mask
            )
            if hasattr(self, "input_B_label") and len(self.input_B_label) > 0:
                self.loss_out_mask_BA = (
                    self.criterionMask(self.real_B_out_mask, self.fake_A_out_mask)
                    * lambda_out_mask
                )
            else:
                self.loss_out_mask_BA = 0
            self.loss_G += self.loss_out_mask_AB + self.loss_out_mask_BA
