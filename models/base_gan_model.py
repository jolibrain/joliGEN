import copy
import os
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torchviz import make_dot

# for FID
from data.base_dataset import get_transform
from util.diff_aug import DiffAugment
from util.discriminator import DiscriminatorInfo

# for D accuracy
from util.image_pool import ImagePool
from util.network_group import NetworkGroup
from util.util import save_image, tensor2im

from . import gan_networks
from .base_model import BaseModel

from .modules.projected_d.discriminator import (
    ProjectedDiscriminator,
    TemporalProjectedDiscriminator,
)

# For D loss computing
from .modules import loss
from .modules.sam.sam_inference import (
    init_sam_net,
    load_mobile_sam_weight,
    load_sam_weight,
    predict_sam,
)
from .modules.utils import download_midas_weight, get_scheduler, predict_depth


class BaseGanModel(BaseModel):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def modify_commandline_options(parser, is_train=True):
        """Configures options specific for CUT model"""

        parser.add_argument(
            "--alg_gan_lambda",
            type=float,
            default=1.0,
            help="weight for GAN loss：GAN(G(X))",
        )

        return parser

    def __init__(self, opt, rank):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """

        super().__init__(opt, rank)

        if hasattr(opt, "fs_light"):
            self.fs_light = opt.fs_light

        if opt.dataaug_diff_aug_policy != "":
            self.diff_augment = DiffAugment(
                opt.dataaug_diff_aug_policy, opt.dataaug_diff_aug_proba
            )

        self.objects_to_update = []

        if self.opt.dataaug_APA:
            self.visual_names.append(["APA_img"])

        if self.opt.train_temporal_criterion or "temporal" in opt.D_netDs:
            self.use_temporal = True
        else:
            self.use_temporal = False

        if self.use_temporal:
            visual_names_temporal_real_A = []
            visual_names_temporal_real_B = []
            visual_names_temporal_fake_B = []
            visual_names_temporal_fake_A = []
            for i in range(self.opt.data_temporal_number_frames):
                visual_names_temporal_real_A.append("temporal_real_A_" + str(i))
                visual_names_temporal_real_B.append("temporal_real_B_" + str(i))
            for i in range(self.opt.data_temporal_number_frames):
                visual_names_temporal_fake_B.append("temporal_fake_B_" + str(i))
                if "cycle_gan" in self.opt.model_type:
                    visual_names_temporal_fake_A.append("temporal_fake_A_" + str(i))

            self.visual_names.append(visual_names_temporal_real_A)
            self.visual_names.append(visual_names_temporal_real_B)
            self.visual_names.append(visual_names_temporal_fake_B)
            if "cycle_gan" in self.opt.model_type:
                self.visual_names.append(visual_names_temporal_fake_A)

        if "depth" in opt.D_netDs:
            self.use_depth = True
            self.netfreeze_depth = download_midas_weight(self.opt.model_depth_network)
        else:
            self.use_depth = False

        if "sam" in opt.D_netDs or opt.data_refined_mask:
            self.use_sam = True
            self.netfreeze_sam, self.predictor_sam = init_sam_net(
                opt.model_type_sam, self.opt.D_weight_sam, self.device
            )
        else:
            self.use_sam = False

        # Define loss functions
        losses_G = ["G_tot"]
        losses_D = ["D_tot"]

        if self.opt.train_temporal_criterion:
            self.criterionTemporal = torch.nn.MSELoss()
            losses_G.append("G_temporal_criterion")

        self.loss_names_G = losses_G
        self.loss_names_D = losses_D

        self.loss_functions_G = ["compute_G_loss_GAN"]
        self.forward_functions = ["forward_GAN"]

        if self.opt.train_semantic_mask:
            self.loss_functions_G.append("compute_G_loss_semantic_mask")
            self.forward_functions.append("forward_semantic_mask")

        if self.opt.train_semantic_cls:
            self.loss_functions_G.append("compute_G_loss_semantic_cls")
            self.forward_functions.append("forward_semantic_cls")

        if self.opt.train_temporal_criterion:
            self.loss_functions_G.append("compute_temporal_criterion_loss")

    def init_semantic_cls(self, opt):
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>

        super().init_semantic_cls(opt)

    def init_semantic_mask(self, opt):
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>

        super().init_semantic_mask(opt)

    def forward_GAN(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_A_pool.query(self.real_A)
        self.real_B_pool.query(self.real_B)

        if self.opt.output_display_G_attention_masks:
            images, attentions, outputs = self.netG_A.get_attention_masks(self.real_A)
            for i, cur_mask in enumerate(attentions):
                setattr(self, "attention_" + str(i), cur_mask)

            for i, cur_output in enumerate(outputs):
                setattr(self, "output_" + str(i), cur_output)

            for i, cur_image in enumerate(images):
                setattr(self, "image_" + str(i), cur_image)

        if self.opt.data_online_context_pixels > 0:
            bs = self.get_current_batch_size()
            self.mask_context = torch.ones(
                [
                    bs,
                    self.opt.model_input_nc,
                    self.opt.data_crop_size + self.margin,
                    self.opt.data_crop_size + self.margin,
                ],
                device=self.device,
            )

            self.mask_context[
                :,
                :,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
            ] = torch.zeros(
                [
                    bs,
                    self.opt.model_input_nc,
                    self.opt.data_crop_size,
                    self.opt.data_crop_size,
                ],
                device=self.device,
            )

            self.mask_context_vis = torch.nn.functional.interpolate(
                self.mask_context, size=self.real_A.shape[2:]
            )[:, 0]

        if self.use_temporal:
            self.compute_temporal_fake(objective_domain="B")
            if hasattr(self, "netG_B"):
                self.compute_temporal_fake(objective_domain="A")

    def compute_D_accuracy_pred(self, real, fake, netD):
        pred_real = (netD(real).flatten() > 0.5) * 1
        pred_fake = (netD(fake).flatten() > 0.5) * 1

        FP = F.l1_loss(
            pred_fake, torch.zeros(pred_real.shape).to(self.device), reduction="sum"
        )
        TP = F.l1_loss(
            pred_real, torch.zeros(pred_real.shape).to(self.device), reduction="sum"
        )
        TN = F.l1_loss(
            pred_fake, torch.ones(pred_real.shape).to(self.device), reduction="sum"
        )
        FN = F.l1_loss(
            pred_real, torch.ones(pred_real.shape).to(self.device), reduction="sum"
        )

        prec_real = TP / (TP + FP)
        prec_fake = TN / (TN + FN)
        rec_real = TP / (TP + FN)
        rec_fake = TN / (TN + FP)
        acc = (TP + TN) / (TP + TN + FN + FP)

        return prec_real, prec_fake, rec_real, rec_fake, acc

    def compute_fake_val(self, imgs, netG):
        return_imgs = []
        for img in imgs:
            return_imgs.append(netG(img.unsqueeze(0)))
        return torch.cat(return_imgs)

    def compute_D_accuracy(self):
        real_A = torch.cat(self.real_A_pool.get_all())
        real_B = torch.cat(self.real_B_pool.get_all())
        if hasattr(self, "netD_A"):
            fake_A = self.compute_fake_val(real_B, self.netG_B)
            (
                self.prec_real_A,
                self.prec_fake_A,
                self.rec_real_A,
                self.rec_fake_A,
                self.acc_A,
            ) = self.compute_D_accuracy_pred(real_A, fake_A, self.netD_A)

            fake_A_val = self.compute_fake_val(self.real_B_val, self.netG_B)
            (
                self.prec_real_A_val,
                self.prec_fake_A_val,
                self.rec_real_A_val,
                self.rec_fake_A_val,
                self.acc_A_val,
            ) = self.compute_D_accuracy_pred(self.real_A_val, fake_A_val, self.netD_A)

        if hasattr(self, "netD_B") or hasattr(self, "netD"):
            if hasattr(self, "netD_B"):
                netD = self.netD_B
                netG = self.netG_B
            elif hasattr(self, "netD"):
                netD = self.netD
                netG = self.netG

            fake_B = self.compute_fake_val(real_A, netG)
            (
                self.prec_real_B,
                self.prec_fake_B,
                self.rec_real_B,
                self.rec_fake_B,
                self.acc_B,
            ) = self.compute_D_accuracy_pred(real_B, fake_B, netD)

            fake_B_val = self.compute_fake_val(self.real_B_val, netG)
            (
                self.prec_real_B_val,
                self.prec_fake_B_val,
                self.rec_real_B_val,
                self.rec_fake_B_val,
                self.acc_B_val,
            ) = self.compute_D_accuracy_pred(self.real_A_val, fake_B_val, netD)

    def get_current_D_accuracies(self):
        accuracies = OrderedDict()
        names = []
        if hasattr(self, "netD_A"):
            names += ["acc_A", "prec_real_A", "prec_fake_A", "rec_real_A", "rec_fake_A"]
            names += [
                "acc_A_val",
                "prec_real_A_val",
                "prec_fake_A_val",
                "rec_real_A_val",
                "rec_fake_A_val",
            ]
        if hasattr(self, "netD_B") or hasattr(self, "netD"):
            names += ["acc_B", "prec_real_B", "prec_fake_B", "rec_real_B", "rec_fake_B"]
            names += [
                "acc_B_val",
                "prec_real_B_val",
                "prec_fake_B_val",
                "rec_real_B_val",
                "rec_fake_B_val",
            ]
        for name in names:
            if isinstance(name, str):
                accuracies[name] = float(
                    getattr(self, name)
                )  # float(...) works for both scalar tensor and float number
        return accuracies

    def get_current_APA_prob(self):
        current_APA_prob = OrderedDict()
        current_APA_prob["APA_p"] = 0.0
        current_APA_prob["APA_adjust"] = 0.0
        for discriminator_name in self.discriminators_names:
            loss_calculator_name = "D_" + discriminator_name + "_loss_calculator"
            D_loss = getattr(self, loss_calculator_name)
            current_APA_prob["APA_p"] += float(D_loss.adaptive_pseudo_augmentation_p)
            current_APA_prob["APA_adjust"] += float(D_loss.adjust)

        return current_APA_prob

    def compute_D_loss_generic(
        self, netD, domain_img, loss, real_name=None, fake_name=None
    ):
        noisy = ""
        if self.opt.dataaug_D_noise > 0.0:
            noisy = "_noisy"

        context = ""
        if self.opt.data_online_context_pixels > 0:
            context = "_with_context"

        if fake_name is None:
            fake = getattr(self, "fake_" + domain_img + "_pool").query(
                getattr(self, "fake_" + domain_img + context + noisy)
            )
        else:
            fake = getattr(self, fake_name)

        if self.opt.dataaug_APA:
            fake_2 = getattr(self, "fake_" + domain_img + "_pool").get_random(
                fake.shape[0]
            )
            self.APA_img = fake_2
        else:
            fake_2 = None

        if real_name is None:
            real = getattr(self, "real_" + domain_img + context + noisy)
        else:
            real = getattr(self, real_name)

        if self.opt.data_image_bits != 8 and type(netD) == ProjectedDiscriminator:
            fake = fake.expand(-1, 3, -1, -1)
            real = real.expand(-1, 3, -1, -1)
            if fake_2 is not None:
                fake_2 = fake_2.expand(-1, 3, -1, -1)

        with torch.cuda.amp.autocast(enabled=self.with_amp):
            loss = loss.compute_loss_D(netD, real, fake, fake_2)
        return loss

    def compute_D_loss(self):
        """Calculate GAN loss for discriminators"""

        self.loss_D_tot = 0

        for discriminator in self.discriminators:
            if self.niter % discriminator.compute_every == 0:
                domain = discriminator.name.split("_")[1]
                netD = getattr(self, discriminator.name)
                loss = getattr(self, discriminator.loss_type)
                if discriminator.fake_name is not None:
                    fake_name = discriminator.fake_name + "_" + domain
                if discriminator.real_name is not None:
                    real_name = discriminator.real_name + "_" + domain
                else:
                    fake_name = None
                    real_name = None

                loss_value = self.compute_D_loss_generic(
                    netD,
                    domain,
                    loss,
                    fake_name=fake_name,
                    real_name=real_name,
                )

            else:
                loss_value = torch.zeros([], device=self.device)

            loss_name = "loss_" + discriminator.loss_name_D

            setattr(
                self,
                loss_name,
                loss_value,
            )

            self.loss_D_tot += loss_value

    def compute_G_loss_GAN_generic(
        self, netD, domain_img, loss, real_name=None, fake_name=None
    ):
        context = ""
        if self.opt.data_online_context_pixels > 0:
            context = "_with_context"

        if fake_name is None:
            fake = getattr(self, "fake_" + domain_img + context)
        else:
            fake = getattr(self, fake_name)
        if real_name is None:
            real = getattr(self, "real_" + domain_img + context)
        else:
            real = getattr(self, real_name)

        if hasattr(self, "diff_augment"):
            real = self.diff_augment(real)
            fake = self.diff_augment(fake)

            if fake_name is None:
                setattr(self, "fake_" + domain_img + "_aug", fake)
            else:
                setattr(self, fake_name + "_aug", fake)

            if real_name is None:
                setattr(self, "real_" + domain_img + "_aug", real)
            else:
                setattr(self, real_name + "_aug", real)

        if self.opt.data_image_bits != 8 and type(netD) == ProjectedDiscriminator:
            fake = fake.expand(-1, 3, -1, -1)
            real = real.expand(-1, 3, -1, -1)
        loss = loss.compute_loss_G(netD, real, fake)
        return loss

    def compute_G_loss(self):
        self.loss_G_tot = 0
        for loss_function in self.loss_functions_G:
            with torch.cuda.amp.autocast(enabled=self.with_amp):
                getattr(self, loss_function)()

    def compute_G_loss_GAN(self):
        """Calculate GAN losses for generator(s)"""

        for discriminator in self.discriminators:
            if "mask" in discriminator.name:
                continue
            if self.niter % discriminator.compute_every == 0:
                domain = discriminator.name.split("_")[1]
                netD = getattr(self, discriminator.name)
                loss = getattr(self, discriminator.loss_type)
                if discriminator.fake_name is not None:
                    fake_name = discriminator.fake_name + "_" + domain
                if discriminator.real_name is not None:
                    real_name = discriminator.real_name + "_" + domain
                else:
                    fake_name = None
                    real_name = None

                loss_value = self.opt.alg_gan_lambda * self.compute_G_loss_GAN_generic(
                    netD,
                    domain,
                    loss,
                    fake_name=fake_name,
                    real_name=real_name,
                )

            else:
                loss_value = torch.zeros([], device=self.device)

            loss_name = "loss_" + discriminator.loss_name_G

            setattr(
                self,
                loss_name,
                loss_value,
            )

            self.loss_G_tot += loss_value

        if self.opt.train_temporal_criterion:
            self.compute_temporal_criterion_loss()

    # compute_real_fake_with_depth
    def compute_fake_real_with_depth(self, fake_name, real_name):
        fake_depth = predict_depth(
            getattr(self, fake_name), self.netfreeze_depth, self.opt.model_depth_network
        )
        real_depth = predict_depth(
            getattr(self, real_name), self.netfreeze_depth, self.opt.model_depth_network
        )
        fake_depth_interp = torch.nn.functional.interpolate(
            fake_depth.unsqueeze(1),
            size=getattr(self, fake_name).shape[2:],
            mode="bilinear",
        )
        fake_depth_interp = (
            fake_depth_interp - fake_depth_interp.min()
        ) / fake_depth_interp.max()
        setattr(self, "fake_depth_B", fake_depth_interp)
        real_depth_interp = torch.nn.functional.interpolate(
            real_depth.unsqueeze(1),
            size=getattr(self, real_name).shape[2:],
            mode="bilinear",
        )
        real_depth_interp = (
            real_depth_interp - real_depth_interp.min()
        ) / real_depth_interp.max()
        setattr(self, "real_depth_B", real_depth_interp)

    def compute_fake_real_with_sam(self, fake_name, real_name):
        fake_sam = predict_sam(getattr(self, fake_name), self.predictor_sam)
        real_sam = predict_sam(getattr(self, real_name), self.predictor_sam)
        setattr(self, "fake_sam_B", fake_sam)
        setattr(self, "real_sam_B", real_sam)

    def set_discriminators_info(self):
        self.discriminators = []

        for discriminator_name in self.discriminators_names:
            loss_calculator_name = "D_" + discriminator_name + "_loss_calculator"

            if "temporal" in discriminator_name or "projected" in discriminator_name:
                train_gan_mode = "projected"
            elif "vision_aided" in discriminator_name:
                train_gan_mode = "vanilla"
            else:
                train_gan_mode = self.opt.train_gan_mode

            if "projected" in discriminator_name:
                dataaug_D_diffusion = self.opt.dataaug_D_diffusion
                dataaug_D_diffusion_every = self.opt.dataaug_D_diffusion_every
            else:
                dataaug_D_diffusion = False
                dataaug_D_diffusion_every = False

            if "temporal" in discriminator_name:
                setattr(
                    self,
                    loss_calculator_name,
                    loss.DiscriminatorGANLoss(
                        netD=getattr(self, "net" + discriminator_name),
                        device=self.device,
                        dataaug_APA_p=self.opt.dataaug_APA_p,
                        dataaug_APA_target=self.opt.dataaug_APA_target,
                        train_batch_size=self.opt.train_batch_size,
                        dataaug_APA_nimg=self.opt.dataaug_APA_nimg,
                        dataaug_APA_every=self.opt.dataaug_APA_every,
                        dataaug_D_label_smooth=self.opt.dataaug_D_label_smooth,
                        train_gan_mode=train_gan_mode,
                        dataaug_APA=self.opt.dataaug_APA,
                        dataaug_D_diffusion=dataaug_D_diffusion,
                        dataaug_D_diffusion_every=dataaug_D_diffusion_every,
                    ),
                )

                fake_name = "temporal_fake"
                real_name = "temporal_real"
                compute_every = self.opt.D_temporal_every

            else:
                fake_name = None
                real_name = None
                compute_every = 1

                if self.opt.train_use_contrastive_loss_D:
                    loss_calculator = (
                        loss.DiscriminatorContrastiveLoss(
                            netD=getattr(self, "net" + discriminator_name),
                            device=self.device,
                            dataaug_APA_p=self.opt.dataaug_APA_p,
                            dataaug_APA_target=self.opt.dataaug_APA_target,
                            train_batch_size=self.opt.train_batch_size,
                            dataaug_APA_nimg=self.opt.dataaug_APA_nimg,
                            dataaug_APA_every=self.opt.dataaug_APA_every,
                            model_input_nc=self.opt.model_input_nc,
                            train_crop_size=train_crop_size,
                            dataaug_APA=self.opt.dataaug_APA,
                        ),
                    )
                else:
                    loss_calculator = loss.DiscriminatorGANLoss(
                        netD=getattr(self, "net" + discriminator_name),
                        device=self.device,
                        dataaug_APA_p=self.opt.dataaug_APA_p,
                        dataaug_APA_target=self.opt.dataaug_APA_target,
                        train_batch_size=self.opt.train_batch_size,
                        dataaug_APA_nimg=self.opt.dataaug_APA_nimg,
                        dataaug_APA_every=self.opt.dataaug_APA_every,
                        dataaug_D_label_smooth=self.opt.dataaug_D_label_smooth,
                        train_gan_mode=train_gan_mode,
                        dataaug_APA=self.opt.dataaug_APA,
                        dataaug_D_diffusion=dataaug_D_diffusion,
                        dataaug_D_diffusion_every=dataaug_D_diffusion_every,
                    )

                setattr(
                    self,
                    loss_calculator_name,
                    loss_calculator,
                )

            if "depth" in discriminator_name:
                fake_name = "fake_depth"
                real_name = "real_depth"
            elif "mask" in discriminator_name:
                fake_name = "fake_mask"
                real_name = "real_mask"
            elif "sam" in discriminator_name:
                fake_name = "fake_sam"
                real_name = "real_sam"

            self.objects_to_update.append(getattr(self, loss_calculator_name))

            self.discriminators.append(
                DiscriminatorInfo(
                    name="net" + discriminator_name,
                    loss_name_D="D_GAN_" + discriminator_name,
                    loss_name_G="G_GAN_" + discriminator_name,
                    loss_type=loss_calculator_name,
                    fake_name=fake_name,
                    real_name=real_name,
                    compute_every=compute_every,
                )
            )

    # multimodal input latent vector
    def get_z_random(self, batch_size, nz, random_type="gauss"):
        if random_type == "uni":
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == "gauss":
            z = torch.randn(batch_size, nz)
        return z.detach().to(self.device)

    def compute_temporal_criterion_loss_generic(self, domain):
        loss_value = torch.zeros([], device=self.device)
        previous_fake = getattr(self, "temporal_fake_" + domain)[:, 0].clone().detach()

        for i in range(1, self.opt.data_temporal_number_frames):
            next_fake = getattr(self, "temporal_fake_" + domain)[:, i]
            loss_value += self.criterionTemporal(previous_fake, next_fake)
            previous_fake = next_fake.clone().detach()

        return loss_value.mean()

    def compute_temporal_criterion_loss(self):
        self.loss_G_temporal_criterion_B = (
            self.compute_temporal_criterion_loss_generic(domain="B")
            * self.opt.train_temporal_criterion_lambda
        )

        if hasattr(self, "netG_B"):
            self.loss_G_temporal_criterion_A = (
                self.compute_temporal_criterion_loss_generic(domain="A")
            ) * self.opt.train_temporal_criterion_lambda
        else:
            self.loss_G_temporal_criterion_A = torch.zeros([], device=self.device)

        self.loss_G_temporal_criterion = (
            self.loss_G_temporal_criterion_B + self.loss_G_temporal_criterion_A
        )

        self.loss_G_tot += self.loss_G_temporal_criterion

    def compute_G_loss_semantic_cls_generic(self, domain_fake):
        """Calculate semantic class loss for G"""

        domain_real = "A" if domain_fake == "B" else "B"
        direction = domain_real + domain_fake

        if not self.opt.train_cls_regression:
            loss_G_sem_cls = self.criterionCLS(
                getattr(self, "pred_cls_fake_%s" % domain_fake),
                getattr(self, "input_%s_label_cls" % domain_real),
            )
        else:
            loss_G_sem_cls = self.criterionCLS(
                getattr(self, "pred_cls_fake_%s" % domain_fake).squeeze(1),
                getattr(self, "input_%s_label_cls" % domain_real),
            )

        if self.opt.train_sem_idt:
            if self.opt.train_sem_net_output or not hasattr(
                self, "input_%s_label_cls" % domain_fake
            ):
                label_idt = (
                    getattr(self, "gt_pred_f_s_real_%s_max" % domain_fake)
                    .clone()
                    .detach()
                )
            else:
                label_idt = getattr(self, "input_%s_label_cls" % domain_fake)

            loss_G_sem_cls_idt = self.opt.train_sem_cls_lambda * self.criterioncls(
                getattr(self, "pred_cls_idt_%s" % domain_fake), label_idt
            )

        # Check if cls is good enough
        if (
            not hasattr(self, "loss_CLS")
            or self.loss_CLS > self.opt.f_s_semantic_threshold
        ):
            loss_G_sem_cls = 0 * loss_G_sem_cls

            if self.opt.train_sem_idt:
                loss_G_sem_cls_idt = 0 * loss_G_sem_cls_idt

        loss_G_sem_cls *= self.opt.train_sem_cls_lambda

        setattr(self, "loss_G_sem_cls_%s" % direction, loss_G_sem_cls)

        self.loss_G_tot += loss_G_sem_cls

        if self.opt.train_sem_idt:
            setattr(self, "loss_G_sem_cls_idt_%s" % domain_fake, loss_G_sem_cls_idt)
            self.loss_G_tot += loss_G_sem_cls_idt

    def compute_G_loss_semantic_cls(self):
        self.compute_G_loss_semantic_cls_generic(domain_fake="B")
        if hasattr(self, "fake_A"):
            self.compute_G_loss_semantic_cls_generic(domain_fake="A")

    def compute_G_loss_semantic_mask(self):
        self.compute_G_loss_semantic_mask_generic(domain_fake="B")

        if hasattr(self, "fake_A"):
            self.compute_G_loss_semantic_mask_generic(domain_fake="A")

    def compute_G_loss_semantic_mask_generic(self, domain_fake):
        """Calculate semantic mask loss for G"""

        domain_real = "A" if domain_fake == "B" else "B"
        direction = domain_real + domain_fake

        if self.opt.train_mask_for_removal:
            label_fake = torch.zeros_like(self.input_A_label_mask)
        elif self.opt.train_sem_net_output or "mask" in self.opt.D_netDs:
            label_fake = getattr(
                self, "gt_pred_f_s_real_%s_max" % domain_real
            )  # argmax
        else:
            label_fake = getattr(self, "input_%s_label_mask" % domain_real)  # logits

        loss_G_sem_mask = self.opt.train_sem_mask_lambda * self.criterionf_s(
            getattr(self, "pred_f_s_fake_%s" % domain_fake), label_fake
        )

        if self.opt.train_sem_idt:
            if self.opt.train_mask_for_removal:
                label_idt = torch.zeros_like(self.input_A_label_mask)
            elif self.opt.train_sem_net_output or not hasattr(
                self, "input_%s_label_mask" % domain_fake
            ):
                label_idt = getattr(
                    self, "gt_pred_f_s_real_%s_max" % domain_fake
                )  # argmax
            else:
                label_idt = getattr(self, "input_%s_label_mask" % domain_fake)  # logits

            loss_G_sem_mask_idt = self.opt.train_sem_mask_lambda * self.criterionf_s(
                getattr(self, "pred_f_s_idt_%s" % domain_fake), label_idt
            )

        # Check if f_s is good enough
        if (
            not hasattr(self, "loss_f_s")
            or self.loss_f_s > self.opt.f_s_semantic_threshold
        ) and self.opt.f_s_net != "sam":
            loss_G_sem_mask = 0 * loss_G_sem_mask
            if self.opt.train_sem_idt:
                loss_G_sem_mask_idt = 0 * loss_G_sem_mask_idt

        setattr(self, "loss_G_sem_mask_%s" % direction, loss_G_sem_mask)

        self.loss_G_tot += loss_G_sem_mask

        if self.opt.train_sem_idt:
            setattr(self, "loss_G_sem_mask_idt_%s" % domain_fake, loss_G_sem_mask_idt)
            self.loss_G_tot += loss_G_sem_mask_idt

        # Out mask loss
        if hasattr(self, "criterionMask"):
            loss_G_out_mask = (
                self.criterionMask(
                    getattr(self, "real_%s_out_mask" % domain_real),
                    self.fake_B_out_mask,
                )
                * self.opt.train_mask_lambda_out_mask
            )

            setattr(self, "loss_G_out_mask_%s" % direction, loss_G_out_mask)

            self.loss_G_tot += loss_G_out_mask
