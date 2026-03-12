import itertools

import torch

from util.network_group import NetworkGroup

from . import gan_networks
from .b2b_model import B2BModel
from .base_gan_model import BaseGanModel


class B2BGanModel(B2BModel, BaseGanModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = BaseGanModel.modify_commandline_options(parser, is_train=is_train)
        parser = B2BModel.modify_commandline_options(parser, is_train=is_train)
        parser.set_defaults(alg_gan_lambda=0.01)
        return parser

    @staticmethod
    def after_parse(opt):
        opt = B2BModel.after_parse(opt)
        if opt.D_netDs == ["none"]:
            raise ValueError(
                "b2b_gan requires at least one discriminator via --D_netDs."
            )
        if opt.G_netG == "vit_vid":
            raise ValueError(
                "b2b_gan does not support G_netG=vit_vid yet. "
                "Use an image B2B backbone for adversarial training."
            )
        return opt

    def __init__(self, opt, rank):
        B2BModel.__init__(self, opt, rank)
        b2b_model_names = self.model_names
        b2b_visual_names = self.visual_names
        BaseGanModel.__init__(self, self.opt, rank)
        self.model_names = b2b_model_names + self.model_names
        self.visual_names = b2b_visual_names + self.visual_names
        self.visual_names.append(["real_A", "fake_B"])
        self.visual_names.append(["real_B"])
        self.requires_x_pred_for_losses = True

        losses_D = []
        losses_G = ["G_b2b"]

        if self.isTrain:
            self.netDs = gan_networks.define_D(**vars(opt))

            self.discriminators_names = [
                "D_B_" + discriminator_name for discriminator_name in self.netDs.keys()
            ]
            self.model_names += self.discriminators_names

            for discriminator_name, netD in self.netDs.items():
                setattr(self, "netD_B_" + discriminator_name, netD)

            D_parameters = itertools.chain(
                *[
                    getattr(self, "net" + discriminator_name).parameters()
                    for discriminator_name in self.discriminators_names
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
            self.optimizers.append(self.optimizer_D)

            self.group_G.backward_functions = ["compute_b2b_gan_loss"]

            self.group_D = NetworkGroup(
                networks_to_optimize=self.discriminators_names,
                forward_functions=None,
                backward_functions=["compute_D_loss"],
                loss_names_list=["loss_names_D"],
                optimizer=["optimizer_D"],
                loss_backward=["loss_D_tot"],
            )
            self.networks_groups.append(self.group_D)
            self.set_discriminators_info()

            for discriminator in self.discriminators:
                losses_D.append(discriminator.loss_name_D)
                if "mask" in discriminator.name:
                    continue
                losses_G.append(discriminator.loss_name_G)

        self.loss_names_D += losses_D
        self.loss_names_G += losses_G
        self.loss_names = self.loss_names_G + self.loss_names_D

        self.iter_calculator_init()

    def compute_G_loss(self):
        for loss_function in self.loss_functions_G:
            with torch.cuda.amp.autocast(enabled=self.with_amp):
                getattr(self, loss_function)()

    def compute_b2b_gan_loss(self):
        self.compute_b2b_loss()
        self.loss_G_b2b = self.loss_G_tot.clone().detach()
        self.fake_B = self.pred_x
        self.compute_G_loss()
