import copy
import os
from abc import abstractmethod
from collections import OrderedDict

# for FID
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F
from torchviz import make_dot
from tqdm import tqdm

from util.diff_aug import DiffAugment
from util.util import save_image, tensor2im
from util.util import pairs_of_floats, pairs_of_ints, MAX_INT

from .base_model import BaseModel
from .modules.sam.sam_inference import (
    init_sam_net,
    load_mobile_sam_weight,
    load_sam_weight,
    predict_sam_mask,
)
from .modules.utils import download_mobile_sam_weight, download_sam_weight


class BaseDiffusionModel(BaseModel):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    @staticmethod
    def modify_commandline_options_train(parser):
        parser = BaseModel.modify_commandline_options_train(parser)

        parser.add_argument(
            "--alg_diffusion_task",
            default="inpainting",
            choices=["inpainting", "super_resolution", "pix2pix"],
            help="Whether to perform inpainting, super resolution or pix2pix",
        )

        parser.add_argument(
            "--alg_diffusion_lambda_G",
            type=float,
            default=1.0,
            help="weight for supervised loss",
        )

        parser.add_argument(
            "--alg_diffusion_dropout_prob",
            type=float,
            default=0.0,
            help="dropout probability for classifier-free guidance",
        )

        parser.add_argument(
            "--alg_diffusion_cond_image_creation",
            type=str,
            default="y_t",
            choices=[
                "y_t",
                "previous_frame",
                "computed_sketch",
                "low_res",
                "ref",
            ],
            help="how image conditioning is created: either from y_t (no conditioning), previous frame, from computed sketch (e.g. canny), from low res image or from reference image (i.e. image that is not aligned with the ground truth)",
        )

        parser.add_argument(
            "--alg_diffusion_cond_computed_sketch_list",
            nargs="+",
            type=str,
            default=["canny", "hed"],
            help="what primitives to use for random sketch",
            choices=["sketch", "canny", "depth", "hed", "hough", "sam"],
        )

        parser.add_argument(
            "--alg_diffusion_cond_sketch_canny_range",
            type=int,
            nargs="+",
            default=[0, 255 * 3],
            help="range of randomized canny sketch thresholds",
        )

        parser.add_argument(
            "--alg_diffusion_super_resolution_scale",
            type=float,
            default=2.0,
            help="scale for super resolution",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_use_gaussian_filter",
            action="store_true",
            default=False,
            help="whether to apply a Gaussian blur to each SAM masks",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_no_sobel_filter",
            action="store_false",
            default=True,
            help="whether to not use a Sobel filter on each SAM masks",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_no_output_binary_sam",
            action="store_false",
            default=True,
            help="whether to not output binary sketch before Canny",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_redundancy_threshold",
            type=float,
            default=0.62,
            help="redundancy threshold above which redundant masks are not kept",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_sobel_threshold",
            type=float,
            default=0.7,
            help="sobel threshold in %% of gradient magnitude",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_final_canny",
            action="store_true",
            default=False,
            help="whether to perform a Canny edge detection on sam sketch to soften the edges",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_min_mask_area",
            type=float,
            default=0.001,
            help="minimum area in proportion of image size for a mask to be kept",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_max_mask_area",
            type=float,
            default=0.99,
            help="maximum area in proportion of image size for a mask to be kept",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_points_per_side",
            type=int,
            default=16,
            help="number of points per side of image to prompt SAM with (# of prompted points will be points_per_side**2)",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_no_sample_points_in_ellipse",
            action="store_false",
            default=True,
            help="whether to not sample the points inside an ellipse to avoid the corners of the image",
        )

        parser.add_argument(
            "--alg_diffusion_cond_sam_crop_delta",
            type=int,
            default=True,
            help="extend crop's width and height by 2*crop_delta before computing masks",
        )

        parser.add_argument(
            "--alg_diffusion_cond_prob_use_previous_frame",
            type=float,
            default=0.5,
            help="prob to use previous frame as y cond",
        )

        parser.add_argument(
            "--alg_diffusion_cond_embed",
            type=str,
            default="",
            choices=["", "mask", "class", "mask_and_class", "ref"],
            help="whether to use conditioning embeddings to the generator layers, and what type",
        )

        parser.add_argument(
            "--alg_diffusion_cond_embed_dim",
            type=int,
            default=32,
            help="nb of examples processed for inference",
        )

        parser.add_argument(
            "--alg_diffusion_generate_per_class",
            action="store_true",
            help="whether to generate samples of each images",
        )

        parser.add_argument(
            "--alg_diffusion_ref_embed_net",
            type=str,
            default="clip",
            choices=["clip", "imagebind"],
            help="embedding network to use for ref conditioning",
        )
        parser.add_argument(
            "--alg_diffusion_vid_canny_dropout",
            type=pairs_of_floats,
            default=[[]],
            nargs="+",
            help="the range of probabilities for dropping the canny for each frame",
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

        # Define loss functions
        losses_G = ["G_tot"]

        self.loss_names_G = losses_G

        self.loss_functions_G = ["compute_G_loss_diffusion"]
        self.forward_functions = ["forward_diffusion"]

        if opt.data_refined_mask:
            self.use_sam_mask = True
        else:
            self.use_sam_mask = False
        # if "sam" in opt.alg_palette_computed_sketch_list:
        #    self.use_sam_edge = True
        # else:
        #    self.use_sam_edge = False
        # if opt.data_refined_mask or "sam" in opt.alg_palette_computed_sketch_list:
        #    self.freezenet_sam, _ = init_sam_net(
        #        opt.model_type_sam, opt.f_s_weight_sam, self.device
        #    )

    def init_semantic_cls(self, opt):
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>

        super().init_semantic_cls(opt)

    def init_semantic_mask(self, opt):
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>

        super().init_semantic_mask(opt)

    def forward_diffusion(self):
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
