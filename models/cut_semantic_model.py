import numpy as np
import torch
from .cut_model import CUTModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .modules.loss import loss
import torch.nn.functional as F
from util.util import gaussian
from util.iter_calculator import IterCalculator
from util.network_group import NetworkGroup


class CUTSemanticModel(CUTModel):
    """This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific for CUT semantic model"""
        parser = CUTModel.modify_commandline_options(parser, is_train=True)
        return parser

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>

        losses_G = ["sem"]

        losses_CLS = ["CLS"]

        self.loss_names_G += losses_G
        self.loss_names_CLS = losses_CLS
        self.loss_names = (
            self.loss_names_G
            + self.loss_names_CLS
            + self.loss_names_D
            + self.loss_names_F_glanet
        )

        # define networks (both generator and discriminator)
        if self.isTrain:
            self.netCLS = networks.define_C(**vars(opt))

            self.model_names += ["CLS"]

            # define loss functions
            self.criterionCLS = torch.nn.modules.CrossEntropyLoss()

            self.optimizer_CLS = opt.optim(
                opt,
                self.netCLS.parameters(),
                lr=opt.train_sem_lr_f_s,
                betas=(opt.train_beta1, opt.train_beta2),
            )

            if opt.train_sem_regression:
                if opt.train_sem_l1_regression:
                    self.criterionCLS = torch.nn.L1Loss()
                else:
                    self.criterionCLS = torch.nn.modules.MSELoss()
            else:
                self.criterionCLS = torch.nn.modules.CrossEntropyLoss()

            self.optimizers.append(self.optimizer_CLS)

            if self.opt.train_iter_size > 1:
                self.iter_calculator = IterCalculator(self.loss_names)
                for i, cur_loss in enumerate(self.loss_names):
                    if "_avg" not in cur_loss:
                        self.loss_names[i] = cur_loss + "_avg"
                    setattr(self, "loss_" + self.loss_names[i], 0)

            ###Making groups
            self.group_CLS = NetworkGroup(
                networks_to_optimize=["CLS"],
                forward_functions=None,
                backward_functions=["compute_CLS_loss"],
                loss_names_list=["loss_names_CLS"],
                optimizer=["optimizer_CLS"],
                loss_backward=["loss_CLS"],
            )
            self.networks_groups.append(self.group_CLS)

    def set_input_first_gpu(self, data):
        super().set_input_first_gpu(data)
        self.input_A_label = self.input_A_label[: self.bs_per_gpu]
        if hasattr(self, "input_B_label"):
            self.input_B_label = self.input_B_label[: self.bs_per_gpu]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        super().set_input(input)
        if "A_label" in input:
            if not self.opt.train_sem_regression:
                self.input_A_label = input["A_label"].to(self.device)
            else:
                self.input_A_label = (
                    input["A_label"].to(torch.float).to(device=self.device)
                )
        if self.opt.train_sem_cls_B and "B_label" in input:
            if not self.opt.train_sem_regression:
                self.input_B_label = input["B_label"].to(self.device)
            else:
                self.input_B_label = (
                    input["B_label"].to(torch.float).to(device=self.device)
                )

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        super().forward()
        d = 1
        self.pred_real_A = self.netCLS(self.real_A)
        if not self.opt.train_sem_regression:
            _, self.gt_pred_A = self.pred_real_A.max(1)

        self.pred_fake_B = self.netCLS(self.fake_B)
        if not self.opt.train_sem_regression:
            _, self.pfB = self.pred_fake_B.max(1)

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        super().compute_G_loss()
        if not self.opt.train_sem_regression:
            self.loss_sem = self.criterionCLS(self.pred_fake_B, self.input_A_label)
        else:
            self.loss_sem = self.criterionCLS(
                self.pred_fake_B.squeeze(1), self.input_A_label
            )
        if (
            not hasattr(self, "loss_CLS")
            or self.loss_CLS > self.opt.f_s_semantic_threshold
        ):
            self.loss_sem = 0 * self.loss_sem
        self.loss_sem *= self.opt.train_sem_lambda
        self.loss_G += self.loss_sem

    def compute_CLS_loss(self):
        label_A = self.input_A_label
        # forward only real source image through semantic classifier
        pred_A = self.netCLS(self.real_A)
        if not self.opt.train_sem_regression:
            self.loss_CLS = self.opt.train_sem_lambda * self.criterionCLS(
                pred_A, label_A
            )
        else:
            self.loss_CLS = self.opt.train_sem_lambda * self.criterionCLS(
                pred_A.squeeze(1), label_A
            )
        if self.opt.train_sem_cls_B:
            label_B = self.input_B_label
            pred_B = self.netCLS(self.real_B)
            if not self.opt.train_sem_regression:
                self.loss_CLS += self.opt.train_sem_lambda * self.criterionCLS(
                    pred_B, label_B
                )
            else:
                self.loss_CLS += self.opt.train_sem_lambda * self.criterionCLS(
                    pred_B.squeeze(1), label_B
                )
