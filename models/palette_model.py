import copy
import itertools
import math
import random
import warnings

import torch
import torchvision.transforms as T
from torch import nn

from data.online_creation import fill_mask_with_color, fill_mask_with_random
from models.modules.sam.sam_inference import compute_mask_with_sam
from util.iter_calculator import IterCalculator
from util.mask_generation import random_edge_mask
from util.network_group import NetworkGroup

from . import diffusion_networks
from .base_diffusion_model import BaseDiffusionModel
from .modules.loss import MultiScaleDiffusionLoss
from .modules.unet_generator_attn.unet_attn_utils import revert_sync_batchnorm
from einops import rearrange


class PaletteModel(BaseDiffusionModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific to the Palette model"""
        parser = BaseDiffusionModel.modify_commandline_options(
            parser, is_train=is_train
        )

        parser.add_argument(
            "--alg_palette_ddim_num_steps",
            type=int,
            default=10,
            help="number of steps for ddim sampling",
        )

        parser.add_argument(
            "--alg_palette_ddim_eta",
            type=float,
            default=0.5,
            help="eta for ddim sampling variance",
        )

        parser.add_argument(
            "--alg_palette_minsnr",
            action="store_true",
            help="use min-SNR weighting",
        )

        if is_train:
            parser = PaletteModel.modify_commandline_options_train(parser)

        return parser

    @staticmethod
    def modify_commandline_options_train(parser):
        parser = BaseDiffusionModel.modify_commandline_options_train(parser)

        parser.add_argument(
            "--alg_palette_loss",
            type=str,
            default="MSE",
            choices=["L1", "MSE", "multiscale_L1", "multiscale_MSE"],
            help="loss type of the denoising model",
        )

        parser.add_argument(
            "--alg_palette_sampling_method",
            type=str,
            default="ddpm",
            choices=["ddpm", "ddim"],
            help="choose the sampling method between ddpm and ddim",
        )

        return parser

    @staticmethod
    def after_parse(opt):
        if opt.isTrain and opt.alg_diffusion_dropout_prob > 0:
            # we add a class to be the unconditionned one.
            opt.f_s_semantic_nclasses += 1
            opt.cls_semantic_nclasses += 1
        return opt

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        self.task = self.opt.alg_diffusion_task
        if self.task == "super_resolution":
            self.opt.alg_diffusion_cond_image_creation = "low_res"
            self.data_crop_size_low_res = int(
                self.opt.data_crop_size / self.opt.alg_diffusion_super_resolution_scale
            )
            self.transform_lr = T.Resize(
                (self.data_crop_size_low_res, self.data_crop_size_low_res)
            )
            self.transform_hr = T.Resize(
                (self.opt.data_crop_size, self.opt.data_crop_size)
            )

        if opt.isTrain:
            batch_size = self.opt.train_batch_size
        else:
            batch_size = self.opt.test_batch_size

        max_visual_outputs = min(
            max(self.opt.train_batch_size, self.opt.num_test_images),
            self.opt.output_num_images,
        )

        if self.opt.G_netG == "unet_vid":
            max_visual_outputs = (
                self.opt.train_batch_size * self.opt.data_temporal_number_frames
            )

        self.num_classes = max(
            self.opt.f_s_semantic_nclasses, self.opt.cls_semantic_nclasses
        )

        self.use_ref = (
            self.opt.alg_diffusion_cond_image_creation == "ref"
            or "ref" in self.opt.alg_diffusion_cond_embed
            or "ref" in self.opt.G_netG
        )

        # Visuals
        visual_outputs = []
        self.gen_visual_names = [
            "gt_image_",
            "cond_image_",
        ]

        if self.task not in ["super_resolution", "pix2pix"]:
            self.gen_visual_names.extend(["y_t_", "mask_"])

        if (
            self.opt.alg_diffusion_cond_embed != ""
            and self.opt.alg_diffusion_generate_per_class
            and not self.use_ref
        ):
            self.nb_classes_inference = (
                max(self.opt.f_s_semantic_nclasses, self.opt.cls_semantic_nclasses) - 1
            )

            for i in range(self.nb_classes_inference):
                self.gen_visual_names.append("output_" + str(i + 1) + "_")

        elif self.use_ref:
            for i in range(max_visual_outputs):
                self.gen_visual_names.append("ref_" + str(i + 1) + "_")
                self.gen_visual_names.append("output_" + str(i + 1) + "_")

        else:
            self.gen_visual_names.append("output_")

        if self.opt.alg_diffusion_cond_image_creation == "previous_frame":
            self.gen_visual_names.insert(0, "previous_frame_")

        for k in range(max_visual_outputs):
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

        # Define loss functions
        losses_G = ["G_tot"]

        if "multiscale" in self.opt.alg_palette_loss:
            img_size = self.opt.data_crop_size
            img_size_log = math.floor(math.log2(img_size))
            min_size = 32
            min_size_log = math.floor(math.log2(min_size))

            scales = []
            for k in range(min_size_log, img_size_log + 1):
                scales.append(2**k)
                losses_G.append("G_" + str(2**k))
            losses_G.append("G_" + str(img_size))

            self.loss_fn = MultiScaleDiffusionLoss(
                self.opt.alg_palette_loss,
                img_size=self.opt.data_crop_size,
                scales=scales,
            )

        elif self.opt.alg_palette_loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        elif self.opt.alg_palette_loss == "L1":
            self.loss_fn = torch.nn.L1Loss()

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

        self.ddim_num_steps = self.opt.alg_palette_ddim_num_steps
        self.ddim_eta = self.opt.alg_palette_ddim_eta

    def set_input(self, data):
        """must use set_device in tensor"""
        if (
            len(data["A"].to(self.device).shape) == 5
        ):  # we're using temporal successive frames
            self.previous_frame = data["A"].to(self.device)[:, 0]
            if self.opt.G_netG == "unet_vid":
                self.y_t = data["A"].to(self.device)
                self.gt_image = data["B"].to(self.device)
            else:
                self.y_t = data["A"].to(self.device)[:, 1]
                self.gt_image = data["B"].to(self.device)[:, 1]
            if self.task == "inpainting":
                self.previous_frame_mask = data["B_label_mask"].to(self.device)[:, 0]
                ### Note: the sam related stuff should eventually go into the dataloader
                if self.use_sam_mask:
                    if self.opt.data_inverted_mask:
                        temp_mask = data["B_label_mask"].clone()
                        temp_mask[temp_mask > 0] = 2
                        temp_mask[temp_mask == 0] = 1
                        temp_mask[temp_mask == 2] = 0
                    else:
                        temp_mask = data["B_label_mask"].clone()
                    self.mask = compute_mask_with_sam(
                        self.gt_image,
                        temp_mask.to(self.device)[:, 1],
                        self.freezenet_sam,
                        self.device,
                        batched=True,
                    )

                    if self.opt.data_inverted_mask:
                        self.mask[self.mask > 0] = 2
                        self.mask[self.mask == 0] = 1
                        self.mask[self.mask == 2] = 0
                    self.y_t = fill_mask_with_random(self.gt_image, self.mask, -1)
                else:
                    if self.opt.G_netG == "unet_vid":
                        self.mask = data["B_label_mask"].to(self.device)
                    else:
                        self.mask = data["B_label_mask"].to(self.device)[:, 1]
            else:
                self.mask = None
        else:
            if self.task == "inpainting":
                self.y_t = data["A"].to(self.device)
                self.gt_image = data["B"].to(self.device)
                ### Note: the sam related stuff should eventually go into the dataloader
                if self.use_sam_mask:
                    if self.opt.data_inverted_mask:
                        temp_mask = data["B_label_mask"].clone()
                        temp_mask[temp_mask > 0] = 2
                        temp_mask[temp_mask == 0] = 1
                        temp_mask[temp_mask == 2] = 0
                    else:
                        temp_mask = data["B_label_mask"].clone()
                    self.mask = compute_mask_with_sam(
                        self.gt_image,
                        temp_mask.to(self.device),
                        self.freezenet_sam,
                        self.device,
                        batched=True,
                    )
                    if self.opt.data_inverted_mask:
                        self.mask[self.mask > 0] = 2
                        self.mask[self.mask == 0] = 1
                        self.mask[self.mask == 2] = 0
                    self.y_t = fill_mask_with_random(self.gt_image, self.mask, -1)

                else:
                    self.mask = data["B_label_mask"].to(self.device)
            elif self.task == "pix2pix":
                self.y_t = data["A"].to(self.device)
                self.gt_image = data["B"].to(self.device)
                self.mask = None
            else:  # e.g. super-resolution
                self.gt_image = data["A"].to(self.device)
                self.mask = None
        if "B_label_cls" in data:
            self.cls = data["B_label_cls"].to(self.device)
        else:
            self.cls = None

        if self.use_ref:
            self.ref_A = data["ref_A"].to(self.device)

        if self.opt.alg_diffusion_cond_image_creation == "y_t":
            self.cond_image = self.y_t
        elif self.opt.alg_diffusion_cond_image_creation == "previous_frame":
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
            randomize_batch = False  # XXX: unused (beniz)
            if randomize_batch:
                cond_images = []
                for image, mask in zip(self.gt_image, self.mask):
                    fill_img_with_random_sketch = random_edge_mask(
                        fn_list=self.opt.alg_diffusion_cond_computed_sketch_list
                    )
                    if "canny" in fill_img_with_random_sketch.__name__:
                        low = min(self.opt.alg_diffusion_cond_sketch_canny_range)
                        high = max(self.opt.alg_diffusion_cond_sketch_canny_range)
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
            else:  # no randomized batch
                fill_img_with_random_sketch = random_edge_mask(
                    fn_list=self.opt.alg_diffusion_cond_computed_sketch_list
                )
                frame = 0
                if len(self.gt_image.shape) == 5:
                    frame = self.gt_image.shape[1]
                    self.mask = rearrange(self.mask, "b f c h w -> (b f) c h w")
                    self.gt_image = rearrange(self.gt_image, "b f c h w -> (b f) c h w")

                if "canny" in fill_img_with_random_sketch.__name__:
                    low = min(self.opt.alg_diffusion_cond_sketch_canny_range)
                    high = max(self.opt.alg_diffusion_cond_sketch_canny_range)
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
                if frame != 0:
                    self.gt_image = rearrange(
                        self.gt_image, " (b f) c h w -> b f c h w", f=frame
                    )
                    self.mask = rearrange(
                        self.mask, " (b f) c h w -> b f c h w", f=frame
                    )
                    self.cond_image = rearrange(
                        self.cond_image, " (b f) c h w -> b f c h w", f=frame
                    )

        elif self.opt.alg_diffusion_cond_image_creation == "low_res":
            self.cond_image = self.transform_lr(self.gt_image)  # bilinear interpolation
            self.cond_image = self.transform_hr(self.cond_image)  # let's get it back

        elif self.opt.alg_diffusion_cond_image_creation == "ref":
            self.cond_image = self.ref_A

        self.batch_size = self.cond_image.shape[0]

        self.real_A = self.cond_image
        self.real_B = self.gt_image

    def compute_palette_loss(self):
        y_0 = self.gt_image
        y_cond = self.cond_image
        mask = self.mask
        noise = None
        cls = self.cls

        frame = 0
        if mask is not None and len(mask.shape) == 5:
            frame = mask.shape[1]
            mask = rearrange(mask, "b f c h w -> (b f) c h w")

        if self.opt.alg_diffusion_dropout_prob > 0.0:
            drop_ids = (
                torch.rand(mask.shape[0], device=mask.device)
                < self.opt.alg_diffusion_dropout_prob
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
        if frame != 0:
            mask = rearrange(mask, " (b f) c h w -> b f c h w", f=frame)

        if self.use_ref:
            ref = self.ref_A
        else:
            ref = None

        noise, noise_hat, min_snr_loss_weight = self.netG_A(
            y_0=y_0, y_cond=y_cond, noise=noise, mask=mask, cls=cls, ref=ref
        )
        frame = 0
        if len(y_0.shape) == 5:
            frame = y_0.shape[1]
            mask = rearrange(mask, "b f c h w -> (b f) c h w")

        if not self.opt.alg_palette_minsnr:
            min_snr_loss_weight = 1.0

        if mask is not None:
            mask_binary = torch.clamp(mask, min=0, max=1)
            loss = self.loss_fn(
                min_snr_loss_weight * mask_binary * noise,
                min_snr_loss_weight * mask_binary * noise_hat,
            )
        else:
            loss = self.loss_fn(
                min_snr_loss_weight * noise, min_snr_loss_weight * noise_hat
            )

        if isinstance(loss, dict):
            loss_tot = torch.zeros(size=(), device=noise.device)

            for cur_size, cur_loss in loss.items():
                setattr(self, "loss_G_" + cur_size, cur_loss)
                loss_tot += cur_loss

            loss = loss_tot

        self.loss_G_tot = self.opt.alg_diffusion_lambda_G * loss

    def inference(self, nb_imgs, offset=0):
        if hasattr(self.netG_A, "module"):
            netG = self.netG_A.module
        else:
            netG = self.netG_A

        if len(self.opt.gpu_ids) > 1 and self.opt.G_unet_mha_norm_layer == "batchnorm":
            netG = revert_sync_batchnorm(netG)

        # task: inpainting
        if self.task in ["inpainting"]:
            if (
                self.opt.alg_diffusion_cond_embed != ""
                and self.opt.alg_diffusion_generate_per_class
                and not self.use_ref
            ):
                for i in range(self.nb_classes_inference):
                    if "class" in self.opt.alg_diffusion_cond_embed:
                        cur_class = torch.ones_like(self.cls)[:nb_imgs] * (i + 1)
                    else:
                        cur_class = None

                    if "mask" in self.opt.alg_diffusion_cond_embed:
                        cur_mask = self.mask[:nb_imgs].clone().clamp(min=0, max=1) * (
                            i + 1
                        )
                    else:
                        cur_mask = self.mask[:nb_imgs]

                    output, visuals = netG.restoration(
                        y_cond=self.cond_image[:nb_imgs],
                        y_t=self.y_t[:nb_imgs],
                        y_0=self.gt_image[:nb_imgs],
                        mask=cur_mask,
                        sample_num=self.sample_num,
                        cls=cur_class,
                        ddim_num_steps=self.ddim_num_steps,
                        ddim_eta=self.ddim_eta,
                        ref=(
                            self.ref_A[: self.inference_num]
                            if hasattr(self, "ref_A")
                            else None
                        ),
                    )

                    name = "output_" + str(i + 1)
                    setattr(self, name, output)

                    name = "visuals_" + str(i + 1)
                    setattr(self, name, visuals)

                self.fake_B = self.output_1
                self.visuals = self.visuals_1

            elif self.use_ref:
                for i in range(nb_imgs):
                    if self.cls is not None:
                        cls = self.cls[:nb_imgs]
                    else:
                        cls = self.cls

                    if self.mask is not None:
                        mask = self.mask[:nb_imgs]
                    else:
                        mask = self.mask

                    cur_ref = self.ref_A[i : i + 1].expand(nb_imgs, -1, -1, -1)

                    if self.opt.alg_diffusion_cond_image_creation == "ref":
                        y_cond = cur_ref

                    else:
                        y_cond = self.cond_image[:nb_imgs]

                    output, visuals = netG.restoration(
                        y_cond=y_cond,
                        y_t=self.y_t[:nb_imgs],
                        y_0=self.gt_image[:nb_imgs],
                        mask=mask,
                        sample_num=self.sample_num,
                        cls=cls,
                        ref=cur_ref,
                    )

                    name = "ref_" + str(i + 1)
                    setattr(self, name, cur_ref)

                    name = "output_" + str(i + 1)
                    setattr(self, name, output)

                    name = "visuals_" + str(i + 1)
                    setattr(self, name, visuals)

                self.fake_B = self.output_1
                self.visuals = self.visuals_1

            # no class conditioning
            else:
                if self.cls is not None:
                    cls = self.cls[:nb_imgs]
                else:
                    cls = self.cls

                if self.mask is not None:
                    mask = self.mask[:nb_imgs]
                else:
                    mask = self.mask

                self.output, self.visuals = netG.restoration(
                    y_cond=self.cond_image[:nb_imgs],
                    y_t=self.y_t[:nb_imgs],
                    y_0=self.gt_image[:nb_imgs],
                    mask=mask,
                    sample_num=self.sample_num,
                    cls=cls,
                    ddim_num_steps=self.ddim_num_steps,
                    ddim_eta=self.ddim_eta,
                )
                self.fake_B = self.output
        # task: super resolution, pix2pix
        elif self.task in ["super_resolution", "pix2pix"]:
            self.output, self.visuals = netG.restoration(
                y_cond=self.cond_image[:nb_imgs],
                sample_num=self.sample_num,
                cls=None,
            )
            self.fake_B = self.output

        # other tasks
        else:
            self.output, self.visuals = netG.restoration(
                y_cond=self.cond_image[:nb_imgs], sample_num=self.sample_num
            )

        if not self.opt.G_netG == "unet_vid":
            for name in self.gen_visual_names:
                whole_tensor = getattr(self, name[:-1])  # i.e. self.output, ...
                for k in range(min(nb_imgs, self.get_current_batch_size())):
                    cur_name = name + str(offset + k)
                    cur_tensor = whole_tensor[k : k + 1]
                    if "mask" in name:
                        cur_tensor = cur_tensor.squeeze(0)
                    setattr(self, cur_name, cur_tensor)
            for k in range(min(nb_imgs, self.get_current_batch_size())):
                self.fake_B_pool.query(self.visuals[k : k + 1])

        else:
            for name in self.gen_visual_names:
                whole_tensor = getattr(self, name[:-1])  # i.e. self.output, ...
                for bs in range(self.opt.train_batch_size):
                    for k in range(self.opt.data_temporal_number_frames):
                        cur_name = name + str(
                            offset + bs * (self.opt.data_temporal_number_frames) + k
                        )
                        cur_tensor = whole_tensor[bs, k, :, :, :].unsqueeze(0)
                        if "mask" in name:
                            cur_tensor = cur_tensor.squeeze(0)
                        setattr(self, cur_name, cur_tensor)
            for k in range(min(nb_imgs, self.get_current_batch_size())):
                self.fake_B_pool.query(self.visuals[k : k + 1, :, :, :, :])
        if len(self.opt.gpu_ids) > 1 and self.opt.G_unet_mha_norm_layer == "batchnorm":
            netG = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netG)

    def compute_visuals(self, nb_imgs):
        super().compute_visuals(nb_imgs)
        with torch.no_grad():
            self.inference(nb_imgs)

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

        if "class" in self.opt.alg_diffusion_cond_embed:
            dummy_cls = torch.ones(1, device=device, dtype=torch.int64)
        else:
            dummy_cls = None

        dummy_ref = torch.ones(
            1, input_nc, self.opt.data_crop_size, self.opt.data_crop_size, device=device
        )

        dummy_input = (
            dummy_y_0,
            dummy_y_cond,
            dummy_mask,
            dummy_noise,
            dummy_cls,
            dummy_ref,
        )

        return dummy_input
