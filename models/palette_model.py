import copy
import math
import random
import warnings

import torch
import torchvision.transforms as T
from torch import nn


import itertools
import tqdm

from data.online_creation import fill_mask_with_color
from models.modules.sam.sam_inference import compute_mask_with_sam
from util.iter_calculator import IterCalculator
from util.mask_generation import random_edge_mask
from util.network_group import NetworkGroup

from . import diffusion_networks
from .base_diffusion_model import BaseDiffusionModel
from .modules.loss import MultiScaleDiffusionLoss
from .modules.unet_generator_attn.unet_attn_utils import revert_sync_batchnorm


class PaletteModel(BaseDiffusionModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific to the Palette model"""
        parser = BaseDiffusionModel.modify_commandline_options(
            parser, is_train=is_train
        )
        parser.add_argument(
            "--alg_palette_task",
            default="inpainting",
            choices=["inpainting", "super_resolution"],
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
            "--alg_palette_dropout_prob",
            type=float,
            default=0.0,
            help="dropout probability for classifier-free guidance",
        )

        parser.add_argument(
            "--alg_palette_cond_image_creation",
            type=str,
            default="y_t",
            choices=[
                "y_t",
                "previous_frame",
                "computed_sketch",
                "low_res",
            ],
            help="how cond_image is created",
        )

        parser.add_argument(
            "--alg_palette_computed_sketch_list",
            nargs="+",
            type=str,
            default=["canny", "hed"],
            help="what to use for random sketch",
            choices=["sketch", "canny", "depth", "hed", "hough", "sam"],
        )

        parser.add_argument(
            "--alg_palette_sketch_canny_range",
            type=int,
            nargs="+",
            default=[0, 255 * 3],
            help="range for Canny thresholds",
        )

        parser.add_argument(
            "--alg_palette_super_resolution_scale",
            type=float,
            default=2.0,
            help="scale for super resolution",
        )

        parser.add_argument(
            "--alg_palette_sam_use_gaussian_filter",
            action="store_true",
            default=False,
            help="whether to apply a gaussian blur to each SAM masks",
        )

        parser.add_argument(
            "--alg_palette_sam_no_sobel_filter",
            action="store_false",
            default=True,
            help="whether to not use a Sobel filter on each SAM masks",
        )

        parser.add_argument(
            "--alg_palette_sam_no_output_binary_sam",
            action="store_false",
            default=True,
            help="whether to not output binary sketch before Canny",
        )

        parser.add_argument(
            "--alg_palette_sam_redundancy_threshold",
            type=float,
            default=0.62,
            help="redundancy threshold above which redundant masks are not kept",
        )

        parser.add_argument(
            "--alg_palette_sam_sobel_threshold",
            type=float,
            default=0.7,
            help="sobel threshold in % of gradient magintude",
        )

        parser.add_argument(
            "--alg_palette_sam_final_canny",
            action="store_true",
            default=False,
            help="whether to perform a Canny edge detection on sam sketch to soften the edges",
        )

        parser.add_argument(
            "--alg_palette_sam_min_mask_area",
            type=float,
            default=0.001,
            help="minimum area in proportion of image size for a mask to be kept",
        )

        parser.add_argument(
            "--alg_palette_sam_max_mask_area",
            type=float,
            default=0.99,
            help="maximum area in proportion of image size for a mask to be kept",
        )

        parser.add_argument(
            "--alg_palette_sam_points_per_side",
            type=int,
            default=16,
            help="number of points per side of image to prompt SAM with (# of prompted points will be points_per_side**2)",
        )

        parser.add_argument(
            "--alg_palette_sam_no_sample_points_in_ellipse",
            action="store_false",
            default=True,
            help="whether to not sample the points inside an ellipse to avoid the corners of the image",
        )

        parser.add_argument(
            "--alg_palette_sam_crop_delta",
            type=int,
            default=True,
            help="extend crop's width and height by 2*crop_delta before computing masks",
        )

        parser.add_argument(
            "--alg_palette_prob_use_previous_frame",
            type=float,
            default=0.5,
            help="prob to use previous frame as y cond",
        )

        parser.add_argument(
            "--alg_palette_sampling_method",
            type=str,
            default="ddpm",
            choices=["ddpm", "ddim"],
            help="choose the sampling method between ddpm and ddim",
        )

        parser.add_argument(
            "--alg_palette_conditioning",
            type=str,
            default="",
            choices=["", "mask", "class", "mask_and_class"],
            help="whether to use conditioning or not",
        )

        parser.add_argument(
            "--alg_palette_cond_embed_dim",
            type=int,
            default=32,
            help="nb of examples processed for inference",
        )

        parser.add_argument(
            "--alg_palette_generate_per_class",
            action="store_true",
            help="whether to generate samples of each images",
        )

        return parser

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        self.task = self.opt.alg_palette_task
        if self.task == "super_resolution":
            self.opt.alg_palette_cond_image_creation = "low_res"
            self.data_crop_size_low_res = int(
                self.opt.data_crop_size / self.opt.alg_palette_super_resolution_scale
            )
            self.transform_lr = T.Resize(
                (self.data_crop_size_low_res, self.data_crop_size_low_res)
            )
            self.transform_hr = T.Resize(
                (self.opt.data_crop_size, self.opt.data_crop_size)
            )

        if self.opt.alg_palette_inference_num == -1:
            self.inference_num = self.opt.train_batch_size
        else:
            self.inference_num = min(
                self.opt.alg_palette_inference_num, self.opt.train_batch_size
            )

        if self.opt.alg_palette_dropout_prob > 0:
            # we add a class to be the unconditionned one.
            self.opt.f_s_semantic_nclasses += 1
            self.opt.cls_semantic_nclasses += 1

        self.num_classes = max(
            self.opt.f_s_semantic_nclasses, self.opt.cls_semantic_nclasses
        )

        # Visuals
        visual_outputs = []
        self.gen_visual_names = [
            "gt_image_",
            "cond_image_",
        ]

        if self.task != "super_resolution":
            self.gen_visual_names.extend(["y_t_", "mask_"])

        if (
            self.opt.alg_palette_conditioning != ""
            and self.opt.alg_palette_generate_per_class
        ):
            self.nb_classes_inference = (
                max(self.opt.f_s_semantic_nclasses, self.opt.cls_semantic_nclasses) - 1
            )

            for i in range(self.nb_classes_inference):
                self.gen_visual_names.append("output_" + str(i + 1) + "_")
        else:
            self.gen_visual_names.append("output_")

        if self.opt.alg_palette_cond_image_creation == "previous_frame":
            self.gen_visual_names.insert(0, "previous_frame_")

        for k in range(self.inference_num):
            self.visual_names.append([temp + str(k) for temp in self.gen_visual_names])

        self.visual_names.append(visual_outputs)

        if opt.G_nblocks == 9 and "resnet" not in opt.G_netG:
            warnings.warn(
                f"G_nblocks default value {opt.G_nblocks} is too high for palette model, 2 will be used instead."
            )
            opt.G_nblocks = 2

        # Define networks
        self.netG_A = diffusion_networks.define_G(**vars(opt))

        self.model_names = ["G_A"]

        self.model_names_export = ["G_A"]

        G_models = ["G_A"]
        G_parameters = [self.netG_A.parameters()]
        G_parameters = itertools.chain(*G_parameters)

        # Define optimizer
        self.optimizer_G = opt.optim(
            opt,
            G_parameters,
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
            networks_to_optimize=G_models,
            forward_functions=[],
            backward_functions=["compute_palette_loss"],
            loss_names_list=["loss_names_G"],
            optimizer=["optimizer_G"],
            loss_backward=losses_backward,
            networks_to_ema=G_models,
        )
        self.networks_groups.append(self.group_G)

        losses_G = []

        self.loss_names_G += losses_G

        self.loss_names = self.loss_names_G.copy()

        # Itercalculator
        self.iter_calculator_init()

        self.sample_num = 2

    def set_input(self, data):
        """must use set_device in tensor"""

        if (
            len(data["A"].to(self.device).shape) == 5
        ):  # we're using temporal successive frames
            self.previous_frame = data["A"].to(self.device)[:, 0]
            self.y_t = data["A"].to(self.device)[:, 1]
            self.gt_image = data["B"].to(self.device)[:, 1]
            if self.task == "inpainting":
                self.previous_frame_mask = data["B_label_mask"].to(self.device)[:, 0]
                if self.use_sam_mask:
                    self.mask = compute_mask_with_sam(
                        self.gt_image,
                        data["B_label_mask"].to(self.device)[:, 1],
                        self.freezenet_sam,
                        self.device,
                        batched=True,
                    )
                else:
                    self.mask = data["B_label_mask"].to(self.device)[:, 1]
            else:
                self.mask = None
        else:
            if self.task == "inpainting":
                self.y_t = data["A"].to(self.device)
                self.gt_image = data["B"].to(self.device)
                if self.use_sam_mask:
                    self.mask = compute_mask_with_sam(
                        self.gt_image,
                        data["B_label_mask"].to(self.device),
                        self.freezenet_sam,
                        self.device,
                        batched=True,
                    )
                else:
                    self.mask = data["B_label_mask"].to(self.device)
            else:  # e.g. super-resolution
                self.y_t = data["A"].to(self.device)
                self.gt_image = data["A"].to(self.device)
                self.mask = None

        if "B_label_cls" in data:
            self.cls = data["B_label_cls"].to(self.device)
        else:
            self.cls = None

        if self.opt.alg_palette_cond_image_creation == "y_t":
            self.cond_image = self.y_t
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
                self.cond_image = self.cond_image.to(self.device)
        elif self.opt.alg_palette_cond_image_creation == "computed_sketch":
            randomize_batch = False
            if randomize_batch:
                cond_images = []
                for image, mask in zip(self.gt_image, self.mask):
                    fill_img_with_random_sketch = random_edge_mask(
                        fn_list=self.opt.alg_palette_computed_sketch_list
                    )
                    if "canny" in fill_img_with_random_sketch.__name__:
                        low = min(self.opt.alg_palette_sketch_canny_range)
                        high = max(self.opt.alg_palette_sketch_canny_range)
                        batch_cond_image = fill_img_with_random_sketch(
                            image.unsqueeze(0),
                            mask.unsqueeze(0),
                            low_threshold_random=low,
                            high_threshold_random=high,
                        ).squeeze(0)
                    elif "sam" in fill_img_with_random_sketch.__name__:
                        batch_cond_image = fill_img_with_random_sketch(
                            image.unsqueeze(0),
                            mask.unsqueeze(0),
                            sam=self.freezenet_sam,
                            opt=self.opt,
                        ).squeeze(0)
                    else:
                        batch_cond_image = fill_img_with_random_sketch(
                            image.unsqueeze(0), mask.unsqueeze(0)
                        ).squeeze(0)
                    cond_images.append(batch_cond_image)
                self.cond_image = torch.stack(cond_images)
                self.cond_image = self.cond_image.to(self.device)
            else:
                fill_img_with_random_sketch = random_edge_mask(
                    fn_list=self.opt.alg_palette_computed_sketch_list
                )
                if "canny" in fill_img_with_random_sketch.__name__:
                    low = min(self.opt.alg_palette_sketch_canny_range)
                    high = max(self.opt.alg_palette_sketch_canny_range)
                    self.cond_image = fill_img_with_random_sketch(
                        self.gt_image,
                        self.mask,
                        low_threshold_random=low,
                        high_threshold_random=high,
                    )
                elif "sam" in fill_img_with_random_sketch.__name__:
                    self.cond_image = fill_img_with_random_sketch(
                        self.gt_image,
                        self.mask,
                        sam=self.freezenet_sam,
                        opt=self.opt,
                    )
                else:
                    self.cond_image = fill_img_with_random_sketch(
                        self.gt_image, self.mask
                    )

                self.cond_image = self.cond_image.to(self.device)

        elif self.opt.alg_palette_cond_image_creation == "low_res":
            self.cond_image = self.transform_lr(self.gt_image)  # bilinear interpolation
            self.cond_image = self.transform_hr(self.cond_image)  # let's get it back

        self.batch_size = self.cond_image.shape[0]

        self.real_A = self.cond_image
        self.real_B = self.gt_image

    def compute_palette_loss(self):
        y_0 = self.gt_image
        y_cond = self.cond_image
        mask = self.mask
        noise = None
        cls = self.cls

        if self.opt.alg_palette_dropout_prob > 0.0:
            drop_ids = (
                torch.rand(mask.shape[0], device=mask.device)
                < self.opt.alg_palette_dropout_prob
            )
        else:
            drop_ids = None

        if drop_ids is not None:
            if mask is not None:
                # the highest class is the unconditionned one.
                mask = torch.where(
                    drop_ids.reshape(-1, 1, 1, 1).expand(mask.shape),
                    self.num_classes - 1,
                    mask,
                )

            if cls is not None:
                # the highest class is the unconditionned one.
                cls = torch.where(drop_ids, self.num_classes - 1, cls)

        noise, noise_hat = self.netG_A(
            y_0=y_0, y_cond=y_cond, noise=noise, mask=mask, cls=cls
        )

        if mask is not None:
            mask_binary = torch.clamp(mask, min=0, max=1)
            loss = self.loss_fn(mask_binary * noise, mask_binary * noise_hat)
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

        if len(self.opt.gpu_ids) > 1 and self.opt.G_unet_mha_norm_layer == "batchnorm":
            netG = revert_sync_batchnorm(netG)

        # task: inpainting
        if self.task in ["inpainting"]:
            if (
                self.opt.alg_palette_conditioning != ""
                and self.opt.alg_palette_generate_per_class
            ):
                for i in range(self.nb_classes_inference):
                    if "class" in self.opt.alg_palette_conditioning:
                        cur_class = torch.ones_like(self.cls)[: self.inference_num] * (
                            i + 1
                        )
                    else:
                        cur_class = None

                    if "mask" in self.opt.alg_palette_conditioning:
                        cur_class_mask = self.mask[: self.inference_num].clone().clamp(
                            min=0, max=1
                        ) * (i + 1)
                    else:
                        cur_class_mask = None

                    output, visuals = netG.restoration(
                        y_cond=self.cond_image[: self.inference_num],
                        y_t=self.y_t[: self.inference_num],
                        y_0=self.gt_image[: self.inference_num],
                        mask=cur_class_mask[: self.inference_num],
                        sample_num=self.sample_num,
                        cls=cur_class,
                    )

                    name = "output_" + str(i + 1)
                    setattr(self, name, output)

                    name = "visuals_" + str(i + 1)
                    setattr(self, name, visuals)

                self.fake_B = self.output_1
                self.visuals = self.visuals_1

            # no class conditioning
            else:

                self.output, self.visuals = netG.restoration(
                    y_cond=self.cond_image[: self.inference_num],
                    y_t=self.y_t[: self.inference_num],
                    y_0=self.gt_image[: self.inference_num],
                    mask=self.mask[: self.inference_num],
                    sample_num=self.sample_num,
                    cls=self.cls[: self.inference_num],
                )
                self.fake_B = self.output

        # task: super resolution
        elif self.task == "super_resolution":
            self.output, self.visuals = netG.restoration(
                y_cond=self.cond_image[: self.inference_num],
                y_t=self.cond_image[: self.inference_num],
                sample_num=self.sample_num,
                cls=None,
            )
            self.fake_B = self.output

        # other tasks
        else:
            self.output, self.visuals = netG.restoration(
                y_cond=self.cond_image[: self.inference_num], sample_num=self.sample_num
            )

        for name in self.gen_visual_names:
            whole_tensor = getattr(self, name[:-1])
            for k in range(min(self.inference_num, self.get_current_batch_size())):
                cur_name = name + str(k)
                cur_tensor = whole_tensor[k : k + 1]

                if "mask" in name:
                    cur_tensor = cur_tensor.squeeze(0)

                setattr(self, cur_name, cur_tensor)

        for k in range(min(self.inference_num, self.get_current_batch_size())):
            self.fake_B_pool.query(self.visuals[k : k + 1])

        if len(self.opt.gpu_ids) > 1 and self.opt.G_unet_mha_norm_layer == "batchnorm":
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

        dummy_mask = torch.ones(
            1, 1, self.opt.data_crop_size, self.opt.data_crop_size, device=device
        )

        dummy_noise = None

        if "class" in self.opt.alg_palette_conditioning:
            dummy_cls = torch.ones(1, device=device, dtype=torch.int64)
        else:
            dummy_cls = None

        dummy_input = (
            dummy_y_0,
            dummy_y_cond,
            dummy_mask,
            dummy_noise,
            dummy_cls,
        )

        return dummy_input
