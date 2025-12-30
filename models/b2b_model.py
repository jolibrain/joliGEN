import copy
import itertools
import math
import random
import warnings

import torch
import torchvision.transforms as T
import tqdm
from torch import nn

from util.network_group import NetworkGroup
from util.iter_calculator import IterCalculator
from models.modules.diffusion_utils import rearrange_5dto4d_bf, rearrange_4dto5d_bf

from util.mask_generation import random_edge_mask

from . import diffusion_networks
from .base_diffusion_model import BaseDiffusionModel

from piq import DISTS, LPIPS
from torchvision.ops import masks_to_boxes
import torch.nn.functional as F


class B2BModel(BaseDiffusionModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific to the shortcut model"""
        parser = BaseDiffusionModel.modify_commandline_options(
            parser, is_train=is_train
        )
        parser.add_argument(
            "--alg_b2b_num_steps",
            type=int,
            default=1000000,
            help="number of steps before reaching the fully discretized consistency model sampling schedule",
        )

        parser.add_argument(
            "--alg_b2b_perceptual_loss",
            type=str,
            default=[""],
            nargs="*",
            choices=["", "LPIPS", "DISTS"],
            help="optional supervised perceptual loss",
        )
        parser.add_argument(
            "--alg_b2b_lambda_perceptual",
            type=float,
            default=1.0,
            help="weight for LPIPS and DISTS perceptual losses",
        )
        parser.add_argument(
            "--alg_b2b_dists_mean",
            default=[0.485, 0.456, 0.406],  # Imagenet default
            nargs="*",
            type=float,
            help="mean for DISTS perceptual loss",
        )
        parser.add_argument(
            "--alg_b2b_dists_std",
            default=[0.229, 0.224, 0.225],  # Imagenet default
            nargs="*",
            type=float,
            help="std for DISTS perceptual loss",
        )

        parser.add_argument(
            "--alg_b2b_metric_mask",
            action="store_true",
            help="Evaluate the metric PSNR and SSIM on the dilated mask region instead of the cropped image.",
        )
        parser.add_argument(
            "--alg_b2b_denoise_timesteps",
            type=int,
            nargs="+",
            default=[50],
            choices=[1, 2, 4, 8, 16, 32, 64, 128],
            help="number of denoise steps",
        )
        if is_train:
            parser = B2BModel.modify_commandline_options_train(parser)

        return parser

    @staticmethod
    def modify_commandline_options_train(parser):
        parser = BaseDiffusionModel.modify_commandline_options_train(parser)
        return parser

    def after_parse(opt):
        return opt

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        self.task = self.opt.alg_diffusion_task

        if opt.isTrain:
            batch_size = self.opt.train_batch_size
        else:
            batch_size = self.opt.test_batch_size

        self.total_t = (
            self.opt.alg_b2b_num_steps * self.opt.train_batch_size
        )  # scaled with bs

        if (
            self.opt.alg_diffusion_cond_embed != ""
            and self.opt.alg_diffusion_generate_per_class
        ):
            self.nb_classes_inference = (
                max(self.opt.f_s_semantic_nclasses, self.opt.cls_semantic_nclasses) - 1
            )

        # Visuals
        visual_outputs = []
        if (
            self.opt.alg_diffusion_cond_embed != ""
            and self.opt.alg_diffusion_generate_per_class
        ):
            for i in range(self.nb_classes_inference):
                self.gen_visual_names.append("output_" + str(i + 1) + "_")
        elif self.opt.model_type == "b2b":
            base_names = ["gt_image_", "y_t_", "mask_"]
            step_outputs = [
                f"output_{ts}_steps_" for ts in self.opt.alg_b2b_denoise_timesteps
            ]
            self.gen_visual_names = base_names + step_outputs

        else:
            self.gen_visual_names.append("output_")

        if opt.alg_diffusion_cond_image_creation != "y_t":
            self.gen_visual_names.append("cond_image_")
        if self.opt.alg_diffusion_cond_image_creation == "previous_frame":
            self.gen_visual_names.insert(0, "previous_frame_")

        if self.opt.G_netG == "unet_vid":
            max_visual_outputs = (
                self.opt.train_batch_size * self.opt.data_temporal_number_frames
            )
            for k in range(max_visual_outputs):
                self.visual_names.append(
                    [temp + str(k) for temp in self.gen_visual_names]
                )
        else:
            for k in range(self.opt.train_batch_size):
                self.visual_names.append(
                    [temp + str(k) for temp in self.gen_visual_names]
                )
            self.visual_names.append(visual_outputs)

        # Define network
        opt.alg_palette_sampling_method = ""

        opt.alg_diffusion_cond_embed_dim = 256
        self.netG_A = diffusion_networks.define_G(**vars(opt)).to(self.device)
        if opt.isTrain:
            self.netG_A.current_t = max(self.netG_A.current_t, opt.total_iters)
        else:
            self.netG_A.current_t = 0  # placeholder
        print("Setting B2B current_iter to", self.netG_A.current_t)

        self.model_names = ["G_A"]
        self.model_names_export = ["G_A"]

        G_models = ["G_A"]
        G_parameters = [self.netG_A.parameters()]
        G_parameters = itertools.chain(*G_parameters)

        # Define optimizer
        if opt.isTrain:
            self.optimizer_G = opt.optim(
                opt,
                G_parameters,
                lr=opt.train_G_lr,
                betas=(opt.train_beta1, opt.train_beta2),
                weight_decay=opt.train_optim_weight_decay,
                eps=opt.train_optim_eps,
            )
            self.optimizers.append(self.optimizer_G)

        self.loss_names_G = ["G_tot"]
        self.loss_names = self.loss_names_G

        # Make group
        self.networks_groups = []

        losses_backward = ["loss_G_tot"]
        self.group_G = NetworkGroup(
            networks_to_optimize=G_models,
            forward_functions=[],
            backward_functions=["compute_b2b_loss"],
            loss_names_list=["loss_names_G"],
            optimizer=["optimizer_G"],
            loss_backward=losses_backward,
            networks_to_ema=G_models,
        )
        self.networks_groups.append(self.group_G)

        losses_G = []
        if opt.alg_b2b_perceptual_loss != [""]:
            losses_G += ["G_perceptual"]
        self.loss_names_G += losses_G
        self.loss_names = self.loss_names_G.copy()

        # Itercalculator
        self.iter_calculator_init()

        # perceptual losses
        if "LPIPS" in self.opt.alg_b2b_perceptual_loss:
            self.criterionLPIPS = LPIPS().to(self.device)
        if "DISTS" in self.opt.alg_b2b_perceptual_loss:
            self.criterionDISTS = DISTS(
                mean=self.opt.alg_b2b_dists_mean, std=self.opt.alg_b2b_dists_std
            ).to(self.device)

    def set_input(self, data):
        if (
            len(data["A"].to(self.device).shape) == 5
        ) and self.opt.G_netG != "unet_vid":  # we're using temporal successive frames
            self.previous_frame = data["A"].to(self.device)[:, 0]
            self.y_t = data["A"].to(self.device)[:, 1]
            self.gt_image = data["B"].to(self.device)[:, 1]
            self.previous_frame_mask = data["B_label_mask"].to(self.device)[:, 0]
            self.mask = data["B_label_mask"].to(self.device)[:, 1]
        else:
            if self.task == "inpainting":
                # inpainting only
                self.y_t = data["A"].to(self.device)
                self.gt_image = data["B"].to(self.device)
                self.mask = data["B_label_mask"].to(self.device)
            elif self.task == "pix2pix":
                self.y_t = data["A"].to(self.device)
                self.gt_image = data["B"].to(self.device)
                self.mask = None

        if self.opt.alg_diffusion_cond_image_creation == "previous_frame":
            cond_image_list = []
            for cur_frame, cur_mask in zip(
                self.previous_frame.cpu(),
                self.previous_frame_mask.cpu(),
            ):
                if (
                    random.random()
                    < self.opt.alg_diffusion_cond_prob_use_previous_frame
                ):
                    cond_image_list.append(cur_frame.to(self.device))
                else:
                    cond_image_list.append(
                        -1 * torch.ones_like(cur_frame, device=self.device)
                    )

                self.cond_image = torch.stack(cond_image_list)
                self.cond_image = self.cond_image.to(self.device)
        elif self.opt.alg_diffusion_cond_image_creation == "computed_sketch":
            fill_img_with_random_sketch = random_edge_mask(
                fn_list=self.opt.alg_diffusion_cond_computed_sketch_list
            )
            if "canny" in fill_img_with_random_sketch.__name__:
                low = min(self.opt.alg_diffusion_cond_sketch_canny_range)
                high = max(self.opt.alg_diffusion_cond_sketch_canny_range)
                if self.opt.G_netG != "unet_vid":
                    self.cond_image = fill_img_with_random_sketch(
                        self.gt_image,
                        self.mask,
                        low_threshold_random=low,
                        high_threshold_random=high,
                    )
                else:
                    self.mask, self.gt_image = rearrange_5dto4d_bf(
                        self.mask, self.gt_image
                    )

                    random_num = torch.rand(self.gt_image.shape[0])
                    dropout_pro = torch.empty(self.gt_image.shape[0]).uniform_(
                        self.opt.alg_diffusion_vid_canny_dropout[0][0],
                        self.opt.alg_diffusion_vid_canny_dropout[1][0],
                    )
                    canny_frame = (random_num > dropout_pro).int()  # binary canny_frame
                    self.cond_image = fill_img_with_random_sketch(
                        self.gt_image,
                        self.mask,
                        low_threshold_random=low,
                        high_threshold_random=high,
                        select_canny=canny_frame,
                    )

                    self.mask, self.gt_image, self.cond_image = rearrange_4dto5d_bf(
                        self.opt.data_temporal_number_frames,
                        self.mask,
                        self.gt_image,
                        self.cond_image,
                    )

        elif self.task == "pix2pix":
            self.cond_image = self.y_t
        else:  # y_t
            self.cond_image = None

        self.batch_size = self.y_t.shape[0]

        self.real_A = self.y_t
        self.real_B = self.gt_image
        self.label_cls = data.get("A_label_cls", None)
        if self.label_cls is not None:
            self.label_cls = self.label_cls.to(self.device)

    def compute_b2b_loss(self):
        """
        Calls netG_A.forward(), which already runs the model internally.
        """
        y_0 = self.gt_image  # ground truth
        y_cond = self.cond_image  # conditioning
        mask = self.mask

        v_pred, v = self.netG_A(y_0, mask, y_cond, y=self.label_cls)

        # Compute loss (l2 loss)
        loss = ((v_pred - v) ** 2).mean(dim=(1, 2, 3)).mean()
        self.loss_G_tot = loss * self.opt.alg_diffusion_lambda_G

        # perceptual losses, if any
        if "LPIPS" in self.opt.alg_b2b_perceptual_loss:
            if mask_pred_x.size(1) > 3:  # more than 3 channels
                self.loss_G_perceptual_lpips = 0.0
                for c in range(4):  # per channel loss and sum
                    y_0_Bc = y_0[:, c, :, :].unsqueeze(1)
                    mask_pred_Bc = mask_pred_x[:, c, :, :].unsqueeze(1)
                    self.loss_G_perceptual_lpips += self.criterionLPIPS(
                        y_0_Bc, mask_pred_Bc
                    )
            else:
                self.loss_G_perceptual_lpips = torch.mean(
                    self.criterionLPIPS(y_0, mask_pred_x)
                )
        else:
            self.loss_G_perceptual_lpips = 0
        if "DISTS" in self.opt.alg_b2b_perceptual_loss:
            if mask_pred_x.size(1) > 3:  # more than 3 channels
                self.loss_G_perceptual_dists = 0.0
                for c in range(4):  # per channel loss and sum
                    y_0_Bc = y_0[:, c, :, :].unsqueeze(1)
                    mask_pred_Bc = mask_pred_x[:, c, :, :].unsqueeze(1)
                    self.loss_G_perceptual_dists += self.criterionDISTS(
                        y_0_Bc, mask_pred_Bc
                    )
            else:
                self.loss_G_perceptual_dists = self.criterionDISTS(y_0, mask_pred_x)
        else:
            self.loss_G_perceptual_dists = 0

        if self.loss_G_perceptual_lpips > 0 or self.loss_G_perceptual_dists > 0:
            self.loss_G_perceptual = self.opt.alg_b2b_lambda_perceptual * (
                self.loss_G_perceptual_lpips + self.loss_G_perceptual_dists
            )
            self.loss_G_tot += self.loss_G_perceptual

    ###########it is not correct for inference !!!!
    def inference(self, nb_imgs, offset=0):
        if hasattr(self.netG_A, "module"):
            netG = self.netG_A.module
        else:
            netG = self.netG_A

        if len(self.opt.gpu_ids) > 1 and self.opt.G_unet_mha_norm_layer == "batchnorm":
            netG = revert_sync_batchnorm(netG)

        # XXX: inpainting only for now
        if self.mask is not None:
            mask = self.mask[:nb_imgs]
        else:
            mask = self.mask
        # restoration call
        if not self.cond_image is None:
            y_cond = self.cond_image[:nb_imgs]
        else:
            y_cond = None
        if self.task == "pix2pix":  #
            out_shape = list(y_cond.shape)
            out_shape[1] = netG.b2b_model.out_channel
            y_t = torch.zeros(out_shape, device=y_cond.device, dtype=y_cond.dtype)
        else:  # e.g. inpainting
            y_t = self.y_t[:nb_imgs]

        if self.task in ["inpainting"]:
            if (
                self.opt.alg_diffusion_cond_embed != ""
                and self.opt.alg_diffusion_generate_per_class
            ):

                for i in range(self.nb_classes_inference):
                    if "mask" in self.opt.alg_diffusion_cond_embed:
                        cur_mask = self.mask[:nb_imgs].clone().clamp(min=0, max=1) * (
                            i + 1
                        )
                    else:
                        cur_mask = self.mask[:nb_imgs]
                    self.output = netG.restoration(
                        y_t, y_cond, self.opt.alg_b2b_denoise_timesteps, cur_mask
                    )
                    name = "output_" + str(i + 1)
                    setattr(self, name, self.output)
            else:
                self.outputs_per_step = {}
                self.fake_B_dilated_per_step = {}
                self.gt_image_dilated_per_step = {}

                for steps in self.opt.alg_b2b_denoise_timesteps:
                    self.output = netG.restoration(y_t, y_cond, steps, mask)
                    self.outputs_per_step[steps] = self.output
                    name = "output_" + str(steps) + "_steps"
                    setattr(self, name, self.output)

                if self.opt.alg_b2b_metric_mask:
                    if self.opt.G_netG == "unet_vid":
                        B, T, C, H, W = mask.shape
                        (mask_4d,) = rearrange_5dto4d_bf(mask)
                        dilation = 3
                        dilated_mask_4d = F.max_pool2d(
                            mask_4d.float(),
                            kernel_size=2 * dilation + 1,
                            stride=1,
                            padding=dilation,
                        )
                        dilated_mask_4d = (dilated_mask_4d > 0.5).float()
                        (dilated_mask,) = rearrange_4dto5d_bf(T, dilated_mask_4d)
                        dilated_mask_expanded = dilated_mask.expand_as(self.output)
                        for steps, out in self.outputs_per_step.items():
                            dm = dilated_mask.expand_as(out)
                            fake_B_dilated = out * dm
                            gt_image_dilated = self.gt_image * dm
                            B, T, _, H, W = dilated_mask.shape
                            boxes = masks_to_boxes(dilated_mask.view(-1, H, W)).int()
                            fake_crops = [
                                [
                                    fake_B_dilated[b, t, :, y0:y1, x0:x1]
                                    for t, (x0, y0, x1, y1) in enumerate(
                                        boxes[b * T : (b + 1) * T]
                                    )
                                ]
                                for b in range(B)
                            ]
                            gt_crops = [
                                [
                                    gt_image_dilated[b, t, :, y0:y1, x0:x1]
                                    for t, (x0, y0, x1, y1) in enumerate(
                                        boxes[b * T : (b + 1) * T]
                                    )
                                ]
                                for b in range(B)
                            ]
                            self.fake_B_dilated_per_step[steps] = fake_crops
                            self.gt_image_dilated_per_step[steps] = gt_crops
                        self.fake_B = torch.cat(
                            list(self.outputs_per_step.values()), dim=0
                        )
                    else:
                        # image mask: (B, C, H, W)
                        B, C, H, W = mask.shape
                        mask_4d = mask
                        # dilate mask in 4D
                        dilation = 3
                        dilated_mask_4d = F.max_pool2d(
                            mask_4d.float(),
                            kernel_size=2 * dilation + 1,
                            stride=1,
                            padding=dilation,
                        )
                        dilated_mask_4d = (dilated_mask_4d > 0.5).float()  # binarize
                        dilated_mask_expanded = dilated_mask_4d.expand_as(
                            self.output
                        )  # output: (B, C, H, W)
                        for steps, out in self.outputs_per_step.items():
                            dm = dilated_mask_4d.expand_as(out)  # out: (B, C, H, W)
                            fake_B_dilated = out * dm
                            gt_image_dilated = self.gt_image * dm
                            boxes = masks_to_boxes(dilated_mask_4d.squeeze(1)).int()
                            # crop fake
                            fake_crops = [
                                fake_B_dilated[b, :, y0:y1, x0:x1]
                                for b, (x0, y0, x1, y1) in enumerate(boxes)
                            ]
                            # crop gt
                            gt_crops = [
                                gt_image_dilated[b, :, y0:y1, x0:x1]
                                for b, (x0, y0, x1, y1) in enumerate(boxes)
                            ]
                            self.fake_B_dilated_per_step[steps] = fake_crops
                            self.gt_image_dilated_per_step[steps] = gt_crops
                        self.fake_B = torch.cat(
                            list(self.outputs_per_step.values()), dim=0
                        )
        num_steps = len(self.outputs_per_step)
        if num_steps > 0:
            self.fake_B = torch.cat(list(self.outputs_per_step.values()), dim=0)
            if num_steps > 1:
                repeat_dims = [num_steps] + [1] * (self.gt_image.dim() - 1)
                self.gt_image = self.gt_image.repeat(*repeat_dims)
        else:
            self.fake_B = self.output

        self.visuals = self.fake_B  # self.output

        if not self.opt.G_netG == "unet_vid":
            for name in self.gen_visual_names:
                whole_tensor = getattr(self, name[:-1])  # i.e. self.output, ...
                for k in range(min(nb_imgs, self.get_current_batch_size())):
                    cur_name = name + str(offset + k)
                    cur_tensor = whole_tensor[k : k + 1]
                    if "mask" in name:
                        cur_tensor = cur_tensor.squeeze(0)
                    setattr(self, cur_name, cur_tensor)
        else:
            for name in self.gen_visual_names:
                whole_tensor = getattr(self, name[:-1])  # i.e. self.output, ...
                for bs in range(min(nb_imgs, self.get_current_batch_size())):
                    for k in range(self.opt.data_temporal_number_frames):
                        cur_name = name + str(
                            offset + bs * (self.opt.data_temporal_number_frames) + k
                        )
                        cur_tensor = whole_tensor[bs, k, :, :, :].unsqueeze(0)
                        if "mask" in name:
                            cur_tensor = cur_tensor.squeeze(0)
                        setattr(self, cur_name, cur_tensor)

    def compute_visuals(self, nb_imgs):
        super().compute_visuals(nb_imgs)
        with torch.no_grad():
            self.inference(nb_imgs)

    def get_current_visuals(self, nb_imgs, phase="train", test_name=""):
        # hide noisy columns in test mode
        old_visual_names = self.visual_names.copy()
        if phase == "test":
            self.visual_names = [
                [x for x in visual_name if "noisy" not in x]
                for visual_name in self.visual_names
            ]
        x = super().get_current_visuals(nb_imgs, phase, test_name)
        self.visual_names = old_visual_names
        return x
