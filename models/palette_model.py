import torch
import tqdm
import copy
import warnings
import random
import math


from .base_diffusion_model import BaseDiffusionModel
from util.network_group import NetworkGroup
from util.iter_calculator import IterCalculator
from . import diffusion_networks
from util.mask_generation import fill_img_with_sketch, fill_img_with_edges
from data.online_creation import fill_mask_with_color
from .modules.unet_generator_attn.unet_attn_utils import revert_sync_batchnorm
from .modules.loss import MultiScaleDiffusionLoss


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
            choices=["L1", "MSE", "multiscale"],
            help="loss for denoising model",
        )

        parser.add_argument(
            "--alg_palette_inference_num",
            type=int,
            default=-1,
            help="nb of examples processed for inference",
        )

        parser.add_argument(
            "--alg_palette_cond_image_creation",
            type=str,
            default="y_t",
            choices=["y_t", "sketch", "edges", "previous_frame"],
            help="how cond_image is created",
        )

        parser.add_argument(
            "--alg_palette_prob_use_previous_frame",
            type=float,
            default=0.5,
            help="prob to use previous frame as y cond",
        )

        return parser

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        if self.opt.alg_palette_inference_num == -1:
            self.inference_num = self.opt.train_batch_size
        else:
            self.inference_num = min(
                self.opt.alg_palette_inference_num, self.opt.train_batch_size
            )

        # Visuals
        visual_outputs = []
        self.gen_visual_names = ["gt_image_", "cond_image_", "y_t_", "mask_", "output_"]

        if self.opt.alg_palette_cond_image_creation == "previous_frame":
            self.gen_visual_names.insert(0, "previous_frame_")

        for k in range(self.inference_num):
            self.visual_names.append([temp + str(k) for temp in self.gen_visual_names])

        self.visual_names.append(visual_outputs)

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
        elif self.opt.alg_palette_loss == "multiscale":
            self.loss_fn = MultiScaleDiffusionLoss(img_size=self.opt.data_crop_size)

        losses_G = ["G_tot"]

        if self.opt.alg_palette_loss == "multiscale":
            img_size = self.opt.data_crop_size
            img_size_log = math.floor(math.log2(img_size))
            min_size = 32
            min_size_log = math.floor(math.log2(min_size))

            for k in range(min_size_log, img_size_log + 1):
                losses_G.append("G_" + str(2**k))

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

        if (
            len(data["A"].to(self.device).shape) == 5
        ):  # we're using temporal successive frames
            self.previous_frame = data["A"].to(self.device)[:, 0]
            self.y_t = data["A"].to(self.device)[:, 1]
            self.gt_image = data["B"].to(self.device)[:, 1]
            self.previous_frame_mask = data["B_label_mask"].to(self.device)[:, 0]
            self.mask = data["B_label_mask"].to(self.device)[:, 1]
        else:
            self.y_t = data["A"].to(self.device)
            self.gt_image = data["B"].to(self.device)
            self.mask = data["B_label_mask"].to(self.device)

        if self.opt.alg_palette_cond_image_creation == "y_t":
            self.cond_image = self.y_t
        elif self.opt.alg_palette_cond_image_creation == "sketch":
            self.cond_image = fill_img_with_sketch(self.gt_image, self.mask)
        elif self.opt.alg_palette_cond_image_creation == "edges":
            self.cond_image = fill_img_with_edges(self.gt_image, self.mask)
        elif self.opt.alg_palette_cond_image_creation == "previous_frame":
            cond_image_list = []

            for cur_frame, cur_mask in zip(
                self.previous_frame.cpu(),
                self.previous_frame_mask.cpu(),
            ):
                if random.random() < self.opt.alg_palette_prob_use_previous_frame:
                    cond_image_list.append(cur_frame.to(self.device))
                else:
                    cond_image_list.append(
                        -1 * torch.ones_like(cur_frame, device=self.device)
                    )

                self.cond_image = torch.stack(cond_image_list)

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
            temp_mask = torch.clamp(mask, min=0.0, max=1.0)
            loss = self.loss_fn(temp_mask * noise, temp_mask * noise_hat)
        else:
            loss = self.loss_fn(noise, noise_hat)

        if isinstance(loss, dict):
            loss_tot = torch.zeros(size=(), device=noise.device)

            for cur_size, cur_loss in loss.items():

                setattr(self, "loss_G_" + cur_size, cur_loss)
                loss_tot += cur_loss

            loss = loss_tot

        self.loss_G_tot = self.opt.alg_palette_lambda_G * loss

    def inference(self):
        if hasattr(self.netG_A, "module"):
            netG = self.netG_A.module
        else:
            netG = self.netG_A

        if self.opt.G_unet_mha_norm_layer == "batchnorm":
            netG = revert_sync_batchnorm(netG)

        if True or self.task in ["inpainting", "uncropping"]:
            self.output, self.visuals = netG.restoration(
                y_cond=self.cond_image[: self.inference_num],
                y_t=self.y_t[: self.inference_num],
                y_0=self.gt_image[: self.inference_num],
                mask=self.mask[: self.inference_num],
                sample_num=self.sample_num,
            )
        else:
            self.output, self.visuals = netG.restoration(
                y_cond=self.cond_image[: self.inference_num], sample_num=self.sample_num
            )

        for k in range(self.inference_num):
            for name in self.gen_visual_names:
                cur_name = name + str(k)
                cur_tensor = getattr(self, name[:-1])[k : k + 1]

                if "mask" in name:
                    cur_tensor = cur_tensor.squeeze(0)

                setattr(self, cur_name, cur_tensor)

        self.fake_B = self.visuals[-1:]

        if self.opt.G_unet_mha_norm_layer == "batchnorm":
            netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG)

    def compute_visuals(self):
        super().compute_visuals()
        with torch.no_grad():
            self.inference()

    def get_dummy_input(self, device=None):
        if device is None:
            device = self.device

        input_nc = self.opt.model_input_nc
        if self.opt.model_multimodal:
            input_nc += self.opt.train_mm_nz

        dummy_y_0 = torch.randn(
            1, input_nc, self.opt.data_crop_size, self.opt.data_crop_size, device=device
        )

        dummy_y_cond = torch.randn(
            1, input_nc, self.opt.data_crop_size, self.opt.data_crop_size, device=device
        )

        dummy_mask = torch.randn(
            1, 1, self.opt.data_crop_size, self.opt.data_crop_size, device=device
        )

        dummy_noise = None

        dummy_input = (dummy_y_0, dummy_y_cond, dummy_mask, dummy_noise)

        return dummy_input
