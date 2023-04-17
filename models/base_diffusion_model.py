import copy
import os
from abc import abstractmethod
from collections import OrderedDict
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F
from torchviz import make_dot
from tqdm import tqdm

# for FID
from data.base_dataset import get_transform
from util.diff_aug import DiffAugment
from util.network_group import NetworkGroup
from util.util import save_image, tensor2im

from .base_model import BaseModel
from .modules.fid.pytorch_fid.fid_score import (
    _compute_statistics_of_path,
    calculate_frechet_distance,
)
from .modules.utils import download_sam_weights, get_scheduler, predict_sam_mask


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
        if opt.data_refined_mask:
            self.use_sam = True
            self.netfreeze_sam = download_sam_weights()
        else:
            self.use_sam = False

        if opt.data_refined_mask:
            self.use_sam_mask = True
            self.freezenet_sam = download_sam_weights()
            self.freezenet_sam.model.to(self.device)
        elif "sam" in opt.alg_palette_computed_sketch_list:
            self.freezenet_sam = download_sam_weights()
            self.freezenet_sam.model.to(self.device)
            self.use_sam_mask = False
        else:
            self.use_sam_mask = False

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

    def compute_mask_with_sam(self, img, rect_mask):
        # get bbox and cat from rect_mask

        boxes = torch.zeros((rect_mask.shape[0], 4))
        categories = []

        for i in range(rect_mask.shape[0]):
            mask = rect_mask[i].squeeze()
            indices = torch.nonzero(mask)
            x_min = indices[:, 1].min()
            y_min = indices[:, 0].min()
            x_max = indices[:, 1].max()
            y_max = indices[:, 0].max()
            boxes[i] = torch.tensor([x_min, y_min, x_max, y_max])
            categories.append(int(torch.unique(mask)[1]))

        boxes = boxes.to(self.device)
        sam_masks = torch.zeros_like(rect_mask)

        for i in range(rect_mask.shape[0]):
            mask = predict_sam_mask(
                img=img[i],
                bbox=np.array(boxes[i].cpu()),
                predictor=self.freezenet_sam,
                cat=categories[i],
            )
            sam_masks[i] = torch.from_numpy(mask).unsqueeze(0)
        return sam_masks
