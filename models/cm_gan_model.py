import torch

from .cm_model import CMModel
from .base_gan_model import BaseGanModel
from . import gan_networks

from .modules import loss

from util.network_group import NetworkGroup
import itertools


class CMGanModel(CMModel, BaseGanModel):

    def __init__(self, opt, rank):
        CMModel.__init__(self, opt, rank)
        cm_model_names = self.model_names
        cm_visual_names = self.visual_names
        BaseGanModel.__init__(self, self.opt, rank)
        self.model_names = cm_model_names + self.model_names
        self.visual_names = cm_visual_names + self.visual_names
        self.opt.alg_gan_lambda = 0.01
        visual_names_A = ["real_A", "fake_B"]
        visual_names_B = ["real_B"]
        self.visual_names.append(visual_names_A)
        self.visual_names.append(visual_names_B)
        self.lambda1 = 0.6
        self.lambda2 = 1.6

        if self.isTrain:
            # Discriminator(s)
            self.netDs = gan_networks.define_D(**vars(opt))

            self.discriminators_names = [
                "D_B_" + D_name for D_name in self.netDs.keys()
            ]
            self.model_names += self.discriminators_names

            for D_name, netD in self.netDs.items():
                setattr(self, "netD_B_" + D_name, netD)

            if len(self.discriminators_names) > 0:
                D_parameters = itertools.chain(
                    *[
                        getattr(self, "net" + D_name).parameters()
                        for D_name in self.discriminators_names
                    ]
                )
            else:
                D_parameters = getattr(
                    self, "net" + self.discriminators_names[0]
                ).parameters()

            self.optimizer_D = opt.optim(
                opt,
                D_parameters,
                lr=opt.train_D_lr,
                betas=(opt.train_beta1, opt.train_beta2),
                weight_decay=opt.train_optim_weight_decay,
                eps=opt.train_optim_eps,
            )
            self.optimizers.append(self.optimizer_D)

            self.group_G.backward_functions = [
                "compute_cm_gan_loss"
            ]  # modify the backward function

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
            losses_D = []
            losses_G = ["G_cm"]
            for discriminator in self.discriminators:
                losses_D.append(discriminator.loss_name_D)
                if "mask" in discriminator.name:
                    continue
                else:
                    losses_G.append(discriminator.loss_name_G)

        self.loss_names_D += losses_D
        self.loss_names_G += losses_G
        self.loss_names = self.loss_names_G + self.loss_names_D

        # Itercalculator
        self.iter_calculator_init()

    def compute_G_loss(self):
        for loss_function in self.loss_functions_G:
            with torch.cuda.amp.autocast(enabled=self.with_amp):
                getattr(self, loss_function)()

    def lambda_function(self, n, N):
        return self.lambda1 * (n / (N - 1)) ** self.lambda2

    def compute_cm_gan_loss(self):
        self.compute_cm_loss()
        self.loss_G_cm = self.loss_G_tot.clone().detach()
        self.fake_B = self.pred_x
        self.compute_G_loss()
        self.loss_G_cm_gan_tot = self.loss_G_tot
        lambda_gan = self.lambda_function(
            self.opt.total_iters, self.opt.alg_cm_num_steps
        )
        self.compute_D_loss()
        loss_cm_gan_tot = (
            self.loss_G_cm * (1 - lambda_gan)
            + (self.loss_G_cm_gan_tot - self.loss_G_cm + self.loss_D_tot) * lambda_gan
        )
        return loss_cm_gan_tot
