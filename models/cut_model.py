import numpy as np
import torch
import torch.nn.functional as F

from .base_gan_model import BaseGanModel
from . import gan_networks

from .modules import loss
from .modules.NCE.patchnce import PatchNCELoss
from .modules.NCE.monce import MoNCELoss
from .modules.NCE.hDCE import PatchHDCELoss
from .modules.NCE.SRC import SRC_Loss


from util.network_group import NetworkGroup
import util.util as util
from util.util import gaussian

import itertools
import warnings


class CUTModel(BaseGanModel):
    """This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific for CUT model"""
        parser = BaseGanModel.modify_commandline_options(parser, is_train=is_train)
        parser.add_argument(
            "--alg_cut_lambda_NCE",
            type=float,
            default=1.0,
            help="weight for NCE loss: NCE(G(X), X)",
        )
        parser.add_argument(
            "--alg_cut_lambda_SRC",
            type=float,
            default=0.0,
            help="weight for SRC (semantic relation consistency) loss: NCE(G(X), X)",
        )
        parser.add_argument(
            "--alg_cut_HDCE_gamma",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--alg_cut_HDCE_gamma_min",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--alg_cut_nce_idt",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=True,
            help="use NCE loss for identity mapping: NCE(G(Y), Y))",
        )

        parser.add_argument(
            "--alg_cut_MSE_idt",
            action="store_true",
            help="use MSENCE loss for identity mapping: MSE(G(Y), Y))",
        )

        parser.add_argument(
            "--alg_cut_lambda_MSE_idt",
            type=float,
            default=1.0,
            help="weight for MSE identity loss: MSE(G(X), X)",
        )

        parser.add_argument(
            "--alg_cut_nce_layers",
            type=str,
            default="0,4,8,12,16",
            help="compute NCE loss on which layers",
        )
        parser.add_argument(
            "--alg_cut_nce_includes_all_negatives_from_minibatch",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.",
        )
        parser.add_argument(
            "--alg_cut_nce_loss",
            type=str,
            default="monce",
            choices=["patchnce", "monce", "SRC_hDCE"],
            help="CUT contrastice loss",
        )
        parser.add_argument(
            "--alg_cut_netF",
            type=str,
            default="mlp_sample",
            choices=["sample", "mlp_sample", "sample_qsattn", "mlp_sample_qsattn"],
            help="how to downsample the feature map",
        )
        parser.add_argument("--alg_cut_netF_nc", type=int, default=256)
        parser.add_argument(
            "--alg_cut_netF_norm",
            type=str,
            default="instance",
            choices=["instance", "batch", "none"],
            help="instance normalization or batch normalization for F",
        )
        parser.add_argument(
            "--alg_cut_netF_dropout",
            action="store_true",
            help="whether to use dropout with F",
        )
        parser.add_argument(
            "--alg_cut_nce_T", type=float, default=0.07, help="temperature for NCE loss"
        )
        parser.add_argument(
            "--alg_cut_num_patches",
            type=int,
            default=256,
            help="number of patches per layer",
        )
        parser.add_argument(
            "--alg_cut_flip_equivariance",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT",
        )

        return parser

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        # Vanilla cut

        # Images to visualize
        visual_names_A = ["real_A", "fake_B"]
        visual_names_B = ["real_B"]

        if "segformer" in self.opt.G_netG:
            self.opt.alg_cut_nce_layers = "0,1,2,3"
            self.opt.alg_cut_nce_T = 0.2  # default 0.07 is too low, https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Understanding_the_Behaviour_of_Contrastive_Loss_CVPR_2021_paper.pdf for a related study
            warnings.warn(
                "cut with segformer requires nce_layers 0,1,2,3 and nce_T set to 0.2, these values are enforced"
            )
        elif "ittr" in self.opt.G_netG:
            self.opt.alg_cut_nce_layers = ",".join(
                [str(k) for k in range(self.opt.G_nblocks)]
            )
        elif "unet" in self.opt.G_netG:
            self.opt.alg_cut_nce_layers = ",".join(
                str(self.opt.G_nblocks * i - 1)
                for i in range(1, len(self.opt.G_unet_mha_channel_mults) + 1)
            )
        elif "uvit" in self.opt.G_netG:
            self.opt.alg_cut_nce_layers = ",".join(
                str(self.opt.G_nblocks * i - 1)
                for i in range(1, len(self.opt.G_unet_mha_channel_mults) + 1)
            )

        self.nce_layers = [int(i) for i in self.opt.alg_cut_nce_layers.split(",")]

        if opt.alg_cut_nce_idt and self.isTrain:
            visual_names_B += ["idt_B"]
        self.visual_names.insert(0, visual_names_A)
        self.visual_names.insert(1, visual_names_B)

        if self.opt.dataaug_diff_aug_policy != "":
            self.visual_names.append(["fake_B_aug"])
            self.visual_names.append(["real_B_aug"])

        if self.opt.output_display_diff_fake_real:
            self.visual_names.append(["diff_real_A_fake_B"])

        if any("depth" in D_name for D_name in self.opt.D_netDs):
            self.visual_names.append(["real_depth_B", "fake_depth_B"])
        if any("sam" in D_name for D_name in self.opt.D_netDs):
            self.visual_names.append(["real_sam_B", "fake_sam_B"])

        if self.isTrain:
            self.model_names = ["G_A", "F"]
            if self.opt.model_multimodal:
                self.model_names.append("E")
            if self.use_depth:
                self.model_names.append("freeze_depth")
            if self.use_sam:
                self.model_names.append("freeze_sam")

            self.model_names_export = ["G_A"]

        else:  # during test time, only load G
            self.model_names = ["G_A"]

        # define networks (both generator and discriminator)

        # Generator
        if self.opt.model_multimodal:
            tmp_model_input_nc = self.opt.model_input_nc
            self.opt.model_input_nc += self.opt.train_mm_nz
        self.netG_A = gan_networks.define_G(**vars(opt))
        if self.opt.model_multimodal:
            self.opt.model_input_nc = tmp_model_input_nc
        self.netF = gan_networks.define_F(**vars(opt))
        self.netF.set_device(self.device)
        if self.opt.model_multimodal:
            self.netE = gan_networks.define_E(**vars(opt))

        if self.isTrain:
            # Discriminator(s)
            self.netDs = gan_networks.define_D(**vars(opt))

            self.discriminators_names = [
                "D_B_" + D_name for D_name in self.netDs.keys()
            ]
            self.model_names += self.discriminators_names

            for D_name, netD in self.netDs.items():
                setattr(self, "netD_B_" + D_name, netD)

            # define loss functions
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                if opt.alg_cut_nce_loss == "patchnce":
                    self.criterionNCE.append(PatchNCELoss(opt).to(self.device))
                elif opt.alg_cut_nce_loss == "monce":
                    self.criterionNCE.append(MoNCELoss(opt).to(self.device))
                elif opt.alg_cut_nce_loss == "SRC_hDCE":
                    self.criterionNCE.append(PatchHDCELoss(opt).to(self.device))

            if opt.alg_cut_nce_loss == "SRC_hDCE":
                self.criterionR = []
                for nce_layer in self.nce_layers:
                    self.criterionR.append(SRC_Loss(opt).to(self.device))

            if self.opt.alg_cut_MSE_idt:
                self.criterionIdt = torch.nn.L1Loss()

            # Optimizers
            self.optimizer_G = opt.optim(
                opt,
                self.netG_A.parameters(),
                lr=opt.train_G_lr,
                betas=(opt.train_beta1, opt.train_beta2),
                weight_decay=opt.train_optim_weight_decay,
                eps=opt.train_optim_eps,
            )
            if self.opt.model_multimodal:
                self.criterionZ = torch.nn.L1Loss()
                self.optimizer_E = opt.optim(
                    opt,
                    self.netE.parameters(),
                    lr=opt.train_G_lr,
                    betas=(opt.train_beta1, opt.train_beta2),
                    weight_decay=opt.train_optim_weight_decay,
                    eps=opt.train_optim_eps,
                )

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

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if self.opt.model_multimodal:
                self.optimizers.append(self.optimizer_E)

            # Making groups
            self.networks_groups = []

            optimizers = ["optimizer_G", "optimizer_F"]
            losses_backward = ["loss_G_tot"]

            if self.opt.model_multimodal:
                #    optimizers.append("optimizer_E")
                losses_backward.append("loss_G_z")
            self.group_G = NetworkGroup(
                networks_to_optimize=["G_A", "F"],
                forward_functions=["forward"],
                backward_functions=["compute_G_loss"],
                loss_names_list=["loss_names_G"],
                optimizer=optimizers,
                loss_backward=losses_backward,
                networks_to_ema=["G_A"],
            )
            self.networks_groups.append(self.group_G)

            if opt.model_multimodal:
                self.group_E = NetworkGroup(
                    networks_to_optimize=["E"],
                    forward_functions=["forward_E"],
                    backward_functions=["compute_E_loss"],
                    loss_names_list=["loss_names_E"],
                    optimizer=["optimizer_E"],
                    loss_backward=["loss_G_z"],
                )
                self.networks_groups.append(self.group_E)

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

        losses_G = ["G_NCE"]
        losses_D = []
        if opt.alg_cut_nce_idt and self.isTrain:
            losses_G += ["G_NCE_Y"]

        if opt.alg_cut_MSE_idt:
            losses_G += ["G_MSE_idt"]

        if opt.model_multimodal and self.isTrain:
            losses_E = ["G_z"]
            losses_G += ["G_z"]

        if self.isTrain:
            for discriminator in self.discriminators:
                losses_D.append(discriminator.loss_name_D)
                if "mask" in discriminator.name:
                    continue
                else:
                    losses_G.append(discriminator.loss_name_G)

        self.loss_names_G += losses_G
        self.loss_names_D += losses_D
        if self.opt.model_multimodal:
            self.loss_names_E = losses_E
            self.loss_names_G += losses_E
        else:
            self.loss_names_E = []

        self.loss_names = self.loss_names_G + self.loss_names_D
        if self.opt.model_multimodal:
            self.loss_names += self.loss_names_E

        # _vis because images with context are too large for visualization, so we resize it to fit into visdom windows
        if self.opt.data_online_context_pixels > 0:
            self.context_visual_names = [
                "real_A_with_context_vis",
                "real_B_with_context_vis",
                "fake_B_with_context_vis",
                "mask_context_vis",
            ]

            self.visual_names.append(self.context_visual_names)

        if self.opt.train_semantic_mask:
            self.init_semantic_mask(opt)
        if self.opt.train_semantic_cls:
            self.init_semantic_cls(opt)

        self.loss_functions_G.append("compute_G_loss_cut")
        self.forward_functions.insert(1, "forward_cut")

        # Itercalculator
        self.iter_calculator_init()

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """

        self.set_input_first_gpu(data)
        if self.opt.isTrain:
            if self.opt.model_multimodal:
                z_random = self.get_z_random(self.real_A.size(0), self.opt.train_mm_nz)
                z_real = z_random.view(z_random.size(0), z_random.size(1), 1, 1).expand(
                    z_random.size(0),
                    z_random.size(1),
                    self.real_A.size(2),
                    self.real_A.size(3),
                )
                real_A_with_z = torch.cat([self.real_A, z_real], 1)
            else:
                real_A_with_z = self.real_A

            feat_temp = self.netG_A.get_feats(real_A_with_z.cpu(), self.nce_layers)
            self.netF.data_dependent_initialize(feat_temp)

            if (
                self.opt.alg_cut_lambda_NCE > 0.0
                and not self.opt.alg_cut_netF == "sample"
            ):
                self.optimizer_F = self.opt.optim(
                    self.opt,
                    self.netF.parameters(),
                    lr=self.opt.train_G_lr,
                    betas=(self.opt.train_beta1, self.opt.train_beta2),
                    weight_decay=self.opt.train_optim_weight_decay,
                    eps=self.opt.train_optim_eps,
                )
                self.optimizers.append(self.optimizer_F)

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        if self.opt.train_semantic_mask:
            self.data_dependent_initialize_semantic_mask(data)

    def data_dependent_initialize_semantic_mask(self, data):
        visual_names_seg_A = ["input_A_label_mask", "gt_pred_f_s_real_A_max", "pfB_max"]

        if hasattr(self, "input_B_label_mask") or self.opt.f_s_net == "sam":
            visual_names_seg_B = ["input_B_label_mask"]
        else:
            visual_names_seg_B = []

        visual_names_seg_B += ["gt_pred_f_s_real_B_max"]
        if self.opt.train_sem_idt:
            visual_names_seg_B += ["pfB_idt_max"]

        if "mask" in self.opt.D_netDs:
            visual_names_seg_B += ["real_mask_B_inv", "fake_mask_B_inv"]
            if self.opt.data_refined_mask:
                visual_names_seg_B += ["label_sam_B"]

        self.visual_names += [visual_names_seg_A, visual_names_seg_B]

        if self.opt.train_mask_out_mask and self.isTrain:
            visual_names_out_mask_A = ["real_A_out_mask", "fake_B_out_mask"]
            self.visual_names += [visual_names_out_mask_A]

    def inference(self):
        self.real = (
            torch.cat((self.real_A, self.real_B), dim=0)
            if self.opt.alg_cut_nce_idt and self.opt.isTrain
            else self.real_A
        )

        if self.opt.model_multimodal:
            self.z_random = self.get_z_random(self.real_A.size(0), self.opt.train_mm_nz)
            z_real = self.z_random.view(
                self.z_random.size(0), self.z_random.size(1), 1, 1
            ).expand(
                self.z_random.size(0),
                self.z_random.size(1),
                self.real.size(2),
                self.real.size(3),
            )
            z_real = torch.cat(
                [z_real, z_real], 0
            )  # accomodates concatenated real_A and real_B
            self.real_with_z = torch.cat([self.real, z_real], 1)
        else:
            self.real_with_z = self.real

        self.fake = self.netG_A(self.real_with_z)

        self.fake_B = self.fake[: self.real_A.size(0)]

    def forward_cut(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.real = (
            torch.cat((self.real_A, self.real_B), dim=0)
            if self.opt.alg_cut_nce_idt and self.opt.isTrain
            else self.real_A
        )
        if self.opt.alg_cut_flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (
                np.random.random() < 0.5
            )
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        if self.opt.model_multimodal:
            self.z_random = self.get_z_random(self.real_A.size(0), self.opt.train_mm_nz)
            z_real = self.z_random.view(
                self.z_random.size(0), self.z_random.size(1), 1, 1
            ).expand(
                self.z_random.size(0),
                self.z_random.size(1),
                self.real.size(2),
                self.real.size(3),
            )
            z_real = torch.cat(
                [z_real, z_real], 0
            )  # accomodates concatenated real_A and real_B
            self.real_with_z = torch.cat([self.real, z_real], 1)
        else:
            self.real_with_z = self.real

        self.fake = self.netG_A(self.real_with_z)

        self.fake_B = self.fake[: self.real_A.size(0)]

        if self.opt.data_online_context_pixels > 0:
            if self.use_temporal:
                self.compute_temporal_fake_with_context(
                    fake_name="temporal_fake_B_0", real_name="temporal_real_A_0"
                )
            self.compute_fake_with_context(fake_name="fake_B", real_name="real_A")

        if self.use_depth:
            self.compute_fake_real_with_depth(fake_name="fake_B", real_name="real_B")
        if self.use_sam:
            self.compute_fake_real_with_sam(fake_name="fake_B", real_name="real_B")

        if "mask" in self.opt.D_netDs:
            self.compute_fake_real_masks()

        if self.opt.alg_cut_nce_idt:
            self.idt_B = self.fake[self.real_A.size(0) :]

        if self.opt.dataaug_D_noise > 0.0:
            context = ""
            if self.opt.data_online_context_pixels > 0:
                context = "_with_context"

            # if self.use_temporal:
            #    names = ["temporal_fake_B_0", "temporal_real_B_0"]
            # else:
            names = ["fake_B", "real_B"]
            for name in names:
                setattr(
                    self,
                    name + context + "_noisy",
                    gaussian(getattr(self, name + context), self.opt.dataaug_D_noise),
                )

        self.diff_real_A_fake_B = self.real_A - self.fake_B

        if self.opt.model_multimodal:
            self.mu2 = self.netE(self.fake_B)

    def forward_E(self):
        if self.opt.model_multimodal:
            self.z_random = self.get_z_random(self.real_A.size(0), self.opt.train_mm_nz)
            z_real = self.z_random.view(
                self.z_random.size(0), self.z_random.size(1), 1, 1
            ).expand(
                self.z_random.size(0),
                self.z_random.size(1),
                self.real_A.size(2),
                self.real_A.size(3),
            )
            real_A_with_z = torch.cat([self.real_A, z_real], 1)
            fake_B = self.netG_A(real_A_with_z)
            self.mu2 = self.netE(fake_B)

    def compute_G_loss_cut(self):
        """Calculate NCE loss for the generator"""

        # Fake losses
        feat_q_pool, feat_k_pool = self.calculate_feats(self.real_A, self.fake_B)

        if self.opt.alg_cut_lambda_SRC > 0.0 or self.opt.alg_cut_nce_loss == "SRC_hDCE":
            self.loss_G_SRC, weight = self.calculate_R_loss(feat_q_pool, feat_k_pool)
        else:
            self.loss_G_SRC = 0.0
            weight = None

        if self.opt.alg_cut_lambda_NCE > 0.0:
            self.loss_G_NCE = self.calculate_NCE_loss(feat_q_pool, feat_k_pool, weight)
        else:
            self.loss_G_NCE = 0.0

        # Identity losses
        feat_q_pool, feat_k_pool = self.calculate_feats(self.real_B, self.idt_B)
        if self.opt.alg_cut_lambda_SRC > 0.0 or self.opt.alg_cut_nce_loss == "SRC_hDCE":
            self.loss_G_SRC_Y, weight = self.calculate_R_loss(feat_q_pool, feat_k_pool)
        else:
            self.loss_G_SRC = 0.0
            weight = None

        if self.opt.alg_cut_nce_idt and self.opt.alg_cut_lambda_NCE > 0.0:
            self.loss_G_NCE_Y = self.calculate_NCE_loss(
                feat_q_pool, feat_k_pool, weight
            )
            loss_NCE_both = (self.loss_G_NCE + self.loss_G_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_G_NCE

        if self.opt.alg_cut_MSE_idt and self.opt.alg_cut_lambda_MSE_idt > 0.0:
            self.loss_G_MSE_idt = self.opt.alg_cut_lambda_MSE_idt * self.criterionIdt(
                self.real_B, self.idt_B
            )
        else:
            self.loss_G_MSE_idt = 0

        self.loss_G_tot += loss_NCE_both + self.loss_G_MSE_idt
        self.compute_E_loss()
        self.loss_G_tot += self.loss_G_z

    def compute_E_loss(self):
        # multimodal loss
        if self.opt.model_multimodal:
            self.loss_G_z = (
                self.criterionZ(self.mu2, self.z_random) * self.opt.train_mm_lambda_z
            )
        else:
            self.loss_G_z = 0

    def calculate_feats(self, src, tgt):
        if hasattr(self.netG_A, "module"):
            netG_A = self.netG_A.module
        else:
            netG_A = self.netG_A
        if self.opt.model_multimodal:
            z_real = self.z_random.view(
                self.z_random.size(0), self.z_random.size(1), 1, 1
            ).expand(
                self.z_random.size(0),
                self.z_random.size(1),
                self.real.size(2),
                self.real.size(3),
            )
            tgt_with_z = torch.cat([tgt, z_real], 1)
            src_with_z = torch.cat([src, z_real], 1)
        else:
            tgt_with_z = tgt
            src_with_z = src
        feat_q = netG_A.get_feats(tgt_with_z, self.nce_layers)

        if self.opt.alg_cut_flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = netG_A.get_feats(src_with_z, self.nce_layers)

        if "qsattn" in self.opt.alg_cut_netF:
            feat_k_pool, sample_ids, attn_mats = self.netF(
                feat_k, self.opt.alg_cut_num_patches, None, None
            )
            feat_q_pool, _, _ = self.netF(
                feat_q, self.opt.alg_cut_num_patches, sample_ids, attn_mats
            )
        else:
            feat_k_pool, sample_ids = self.netF(
                feat_k, self.opt.alg_cut_num_patches, None
            )
            feat_q_pool, _ = self.netF(feat_q, self.opt.alg_cut_num_patches, sample_ids)

        return feat_q_pool, feat_k_pool

    def calculate_NCE_loss(self, feat_q_pool, feat_k_pool, weights):
        if weights is None:
            weights = [None for k in range(len(feat_q_pool))]
        n_layers = len(self.nce_layers)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer, weight in zip(
            feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers, weights
        ):
            loss = (
                crit(
                    feat_q=f_q,
                    feat_k=f_k,
                    current_batch=self.get_current_batch_size(),
                    weight=weight,
                )
                * self.opt.alg_cut_lambda_NCE
            )

            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_R_loss(self, feat_q_pool, feat_k_pool, only_weight=False, epoch=None):
        n_layers = len(self.nce_layers)

        total_SRC_loss = 0.0
        weights = []
        for f_q, f_k, crit, nce_layer in zip(
            feat_q_pool, feat_k_pool, self.criterionR, self.nce_layers
        ):
            loss_SRC, weight = crit(f_q, f_k, only_weight, epoch)
            total_SRC_loss += loss_SRC * self.opt.alg_cut_lambda_SRC
            weights.append(weight)
        return total_SRC_loss / n_layers, weights
