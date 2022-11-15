import os
import copy
import torch
from collections import OrderedDict
from abc import abstractmethod
from .modules.utils import get_scheduler
from torchviz import make_dot
from .base_model import BaseModel

from util.network_group import NetworkGroup

# for FID
from data.base_dataset import get_transform
from .modules.fid.pytorch_fid.fid_score import (
    _compute_statistics_of_path,
    calculate_frechet_distance,
)
from util.util import save_image, tensor2im
import numpy as np
from util.diff_aug import DiffAugment


from inspect import isfunction


import torch.nn.functional as F


from tqdm import tqdm


class BaseDiffusionModel(BaseModel):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

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
