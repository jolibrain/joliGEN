import copy
import hashlib
import math
import os
from contextlib import ExitStack

import torch
import torch.nn.functional as F

from .base_model import BaseModel
from .modules.mat import Discriminator as MATDiscriminator
from .modules.mat import Generator as MATGenerator
from .modules.mat import PerceptualLoss as MATPerceptualLoss
from .modules.mat.torch_utils.ops import conv2d_gradfix


class MATModel(BaseModel):
    G_REG_INTERVAL = 4

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument(
            "--alg_mat_z_dim",
            type=int,
            default=512,
            help="latent z dimensionality for MAT",
        )
        parser.add_argument(
            "--alg_mat_w_dim",
            type=int,
            default=512,
            help="latent w dimensionality for MAT",
        )
        parser.add_argument(
            "--alg_mat_pcp_ratio",
            type=float,
            default=0.1,
            help="weight for the MAT perceptual loss",
        )
        parser.add_argument(
            "--alg_mat_r1_gamma",
            type=float,
            default=10.0,
            help="R1 regularization weight for the MAT discriminator",
        )
        parser.add_argument(
            "--alg_mat_style_mixing_prob",
            type=float,
            default=0.5,
            help="style mixing probability for MAT generator training",
        )
        parser.add_argument(
            "--alg_mat_truncation_psi",
            type=float,
            default=0.5,
            help="truncation psi used for MAT evaluation and validation inference",
        )
        parser.add_argument(
            "--alg_mat_d_reg_every",
            type=int,
            default=16,
            help="interval for lazy discriminator R1 regularization, 0 disables it",
        )
        parser.add_argument(
            "--alg_mat_transformer_lr",
            type=float,
            default=-1.0,
            help="optional learning rate override for MAT transformer parameters",
        )
        parser.add_argument(
            "--alg_mat_ema_kimg",
            type=float,
            default=10.0,
            help="EMA half-life in kimg for the MAT generator",
        )
        parser.add_argument(
            "--alg_mat_ema_rampup",
            type=float,
            default=0.0,
            help="EMA ramp-up factor, 0 disables ramp-up",
        )
        parser.add_argument(
            "--alg_mat_noise_mode_train",
            type=str,
            default="random",
            choices=["random", "const", "none"],
            help="noise mode used by MAT during training",
        )
        parser.add_argument(
            "--alg_mat_noise_mode_eval",
            type=str,
            default="const",
            choices=["random", "const", "none"],
            help="noise mode used by MAT during evaluation",
        )
        parser.add_argument(
            "--alg_mat_mask_class_conditioning",
            action="store_true",
            help="add a raw class-valued mask channel as extra MAT generator conditioning",
        )
        parser.add_argument(
            "--alg_mat_motion",
            action="store_true",
            help="enable MAT motion conditioning on self_supervised_vid_mask_online windows",
        )
        parser.add_argument(
            "--alg_mat_motion_max_frames",
            type=int,
            default=8,
            help="maximum temporal window supported by the MAT motion module",
        )
        parser.add_argument(
            "--alg_mat_motion_num_attention_heads",
            type=int,
            default=8,
            help="number of attention heads used by the MAT motion module",
        )
        parser.add_argument(
            "--alg_mat_motion_num_transformer_blocks",
            type=int,
            default=2,
            help="number of temporal transformer blocks used by the MAT motion module",
        )

        parser.set_defaults(
            D_netDs=["none"],
            train_G_ema=True,
            train_beta1=0.0,
            train_beta2=0.99,
            train_G_lr=0.001,
            train_D_lr=0.001,
        )

        return parser

    @staticmethod
    def after_parse(opt):
        if opt.alg_mat_motion:
            if opt.data_dataset_mode != "self_supervised_vid_mask_online":
                raise ValueError(
                    "MAT motion is only supported with self_supervised_vid_mask_online"
                )
            if opt.data_temporal_number_frames < 2:
                raise ValueError(
                    "alg_mat_motion requires data_temporal_number_frames >= 2"
                )
            if opt.alg_mat_motion_max_frames < opt.data_temporal_number_frames:
                raise ValueError(
                    "alg_mat_motion_max_frames must be >= data_temporal_number_frames"
                )
        else:
            supported_dataset_modes = {
                "self_supervised_labeled_mask",
                "self_supervised_labeled_mask_online",
                "self_supervised_labeled_mask_ref",
                "self_supervised_labeled_mask_online_ref",
            }
            if opt.data_dataset_mode not in supported_dataset_modes:
                raise ValueError(
                    "MAT is only supported with self-supervised labeled-mask inpainting datasets"
                )

        if opt.train_beta1 == 0.9:
            opt.train_beta1 = 0.0
        if opt.train_beta2 == 0.999:
            opt.train_beta2 = 0.99
        if opt.train_G_lr == 0.0002:
            opt.train_G_lr = 0.001
        if opt.train_D_lr == 0.0001:
            opt.train_D_lr = 0.001
        if not opt.train_G_ema:
            opt.train_G_ema = True

        if opt.data_direction != "AtoB":
            raise ValueError("MAT only supports AtoB inpainting training")
        if opt.model_input_nc != 3 or opt.model_output_nc != 3:
            raise ValueError("MAT currently supports RGB inputs and outputs only")
        if opt.data_crop_size not in {256, 512}:
            raise ValueError("MAT currently supports data_crop_size 256 or 512 only")
        if opt.data_online_context_pixels != 0:
            raise ValueError("MAT does not support context-padded crops")
        if opt.model_multimodal:
            raise ValueError("MAT does not support multimodal training")
        if opt.train_semantic_mask or opt.train_semantic_cls:
            raise ValueError("MAT does not support semantic auxiliary heads")
        if opt.train_temporal_criterion:
            raise ValueError("MAT does not support temporal training")
        if opt.train_export_jit:
            raise ValueError("MAT does not support joliGEN generator export")
        if opt.D_netDs == ["projected_d", "basic"]:
            opt.D_netDs = ["none"]
        elif opt.D_netDs != ["none"]:
            raise ValueError("MAT uses its own discriminator and does not use D_netDs")
        if any(
            loss_name for loss_name in getattr(opt, "alg_cut_supervised_loss", [""])
        ):
            raise ValueError("MAT does not use alg_cut_supervised_loss")
        if opt.alg_mat_d_reg_every < 0:
            raise ValueError("alg_mat_d_reg_every must be non-negative")
        if opt.alg_mat_ema_kimg <= 0:
            raise ValueError("alg_mat_ema_kimg must be strictly positive")
        if opt.alg_mat_ema_rampup < 0:
            raise ValueError("alg_mat_ema_rampup must be non-negative")
        if opt.alg_mat_mask_class_conditioning and opt.f_s_semantic_nclasses < 2:
            raise ValueError(
                "alg_mat_mask_class_conditioning requires f_s_semantic_nclasses >= 2"
            )
        if opt.alg_mat_motion_max_frames < 2:
            raise ValueError("alg_mat_motion_max_frames must be >= 2")
        if opt.alg_mat_motion_num_attention_heads <= 0:
            raise ValueError("alg_mat_motion_num_attention_heads must be > 0")
        if opt.alg_mat_motion_num_transformer_blocks <= 0:
            raise ValueError("alg_mat_motion_num_transformer_blocks must be > 0")

        return opt

    def __init__(self, opt, rank):
        super().__init__(opt, rank)

        if opt.isTrain:
            max_visual_outputs = min(
                max(self.opt.train_batch_size, getattr(self.opt, "num_test_images", 0)),
                self.opt.output_num_images,
            )
        else:
            max_visual_outputs = min(
                self.opt.test_batch_size, self.opt.output_num_images
            )

        self.gen_visual_names = ["gt_image_", "y_t_", "mask_", "output_"]
        if opt.alg_mat_motion:
            self.gen_visual_names.insert(0, "previous_frame_")
        for k in range(max_visual_outputs):
            self.visual_names.append([name + str(k) for name in self.gen_visual_names])

        self.model_names = ["G_A", "D_A"] if opt.isTrain else ["G_A"]
        self.model_names_export = []
        self.networks_groups = []

        self.netG_A = MATGenerator(
            z_dim=opt.alg_mat_z_dim,
            c_dim=0,
            w_dim=opt.alg_mat_w_dim,
            img_resolution=opt.data_crop_size,
            img_channels=opt.model_output_nc,
            synthesis_kwargs={
                "mask_class_channels": 1
                if opt.alg_mat_mask_class_conditioning
                else 0,
                "motion_enabled": opt.alg_mat_motion,
                "motion_max_frames": opt.alg_mat_motion_max_frames,
                "motion_num_attention_heads": opt.alg_mat_motion_num_attention_heads,
                "motion_num_transformer_blocks": opt.alg_mat_motion_num_transformer_blocks,
            },
        )
        self._configure_motion_finetuning()

        if opt.isTrain:
            batch_per_gpu = opt.train_batch_size
            if len(opt.gpu_ids) > 1:
                batch_per_gpu = max(1, batch_per_gpu // len(opt.gpu_ids))
            mbstd_group_size = max(1, min(4, batch_per_gpu))
            self.netD_A = MATDiscriminator(
                c_dim=0,
                img_resolution=opt.data_crop_size,
                img_channels=opt.model_output_nc,
                mbstd_group_size=mbstd_group_size,
            )
            try:
                self.criterionMAT = MATPerceptualLoss(
                    layer_weights=dict(conv4_4=1 / 4, conv5_4=1 / 2)
                ).to(self.device)
            except Exception as exc:
                raise RuntimeError(
                    "MAT perceptual loss initialization failed. Ensure torchvision "
                    "can access pretrained VGG weights or cache them locally."
                ) from exc

            g_lr = opt.train_G_lr
            g_betas = (opt.train_beta1, opt.train_beta2)
            if self.G_REG_INTERVAL > 0:
                g_mb_ratio = self.G_REG_INTERVAL / (self.G_REG_INTERVAL + 1)
                g_lr *= g_mb_ratio
                g_betas = tuple(beta**g_mb_ratio for beta in g_betas)

            self.optimizer_G = opt.optim(
                opt,
                self._build_generator_param_groups(),
                lr=g_lr,
                betas=g_betas,
                weight_decay=opt.train_optim_weight_decay,
                eps=opt.train_optim_eps,
            )

            d_lr = opt.train_D_lr
            d_betas = (opt.train_beta1, opt.train_beta2)
            if opt.alg_mat_d_reg_every > 0:
                mb_ratio = opt.alg_mat_d_reg_every / (opt.alg_mat_d_reg_every + 1)
                d_lr *= mb_ratio
                d_betas = tuple(beta**mb_ratio for beta in d_betas)

            self.optimizer_D = opt.optim(
                opt,
                self.netD_A.parameters(),
                lr=d_lr,
                betas=d_betas,
                weight_decay=opt.train_optim_weight_decay,
                eps=opt.train_optim_eps,
            )
            self.optimizers.extend([self.optimizer_G, self.optimizer_D])

            self.loss_names_G = ["G_adv", "G_adv_stg1", "G_pcp", "G_l1", "G_tot"]
            self.loss_names_D = [
                "D_fake",
                "D_fake_stg1",
                "D_real",
                "D_real_stg1",
                "D_r1",
                "D_r1_stg1",
                "D_tot",
            ]
            self.loss_names = self.loss_names_G + self.loss_names_D
            for loss_name in self.loss_names:
                setattr(
                    self, "loss_" + loss_name, torch.zeros(size=(), device=self.device)
                )
            self.iter_calculator_init()

        self.mask = None
        self.mask_seq = None
        self.mask_class = None
        self.mask_class_seq = None
        self.mask_keep_seq = None
        self.previous_frame = None
        self.real_A_seq = None
        self.real_B_seq = None
        self.gt_image = None
        self.y_t = None
        self.output = None
        self.fake_B = None
        self.fake_B_stg1 = None
        self.eval_seeds = []

    def _ensure_rgb_output(self, tensor, name):
        expected_channels = self.opt.model_output_nc
        if tensor.ndim != 4:
            raise RuntimeError(
                f"MAT {name} must be a 4D tensor, got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != expected_channels:
            raise RuntimeError(
                f"MAT {name} must have {expected_channels} channels, "
                f"got shape {tuple(tensor.shape)}"
            )
        return tensor

    def _build_generator_param_groups(self):
        transformer_lr = self.opt.alg_mat_transformer_lr
        if transformer_lr <= 0:
            return [param for param in self.netG_A.parameters() if param.requires_grad]

        base_params = []
        transformer_params = []
        for name, param in self.netG_A.named_parameters():
            if not param.requires_grad:
                continue
            if (
                "tran" in name
                or "Tran" in name
                or "motion_module" in name
            ):
                transformer_params.append(param)
            else:
                base_params.append(param)

        if len(transformer_params) == 0:
            return [param for param in self.netG_A.parameters() if param.requires_grad]

        param_groups = []
        if len(base_params) > 0:
            param_groups.append({"params": base_params})
        param_groups.append({"params": transformer_params, "lr": transformer_lr})
        return param_groups

    def _configure_motion_finetuning(self):
        if not self.opt.alg_mat_motion:
            return

        self.set_requires_grad(self.netG_A, False)
        netG = self._unwrap_net(self.netG_A)
        trainable_modules = [
            netG.synthesis.motion_module,
            netG.synthesis.to_square,
            netG.synthesis.to_style,
            netG.synthesis.dec,
        ]
        self.set_requires_grad(trainable_modules, True)

    def _set_generator_trainable(self, enabled):
        if not self.opt.alg_mat_motion:
            self.set_requires_grad(self.netG_A, enabled)
            return

        self.set_requires_grad(self.netG_A, False)
        if enabled:
            netG = self._unwrap_net(self.netG_A)
            self.set_requires_grad(
                [
                    netG.synthesis.motion_module,
                    netG.synthesis.to_square,
                    netG.synthesis.to_style,
                    netG.synthesis.dec,
                ],
                True,
            )

    def _motion_sequence_length(self):
        return max(2, self.opt.data_temporal_number_frames)

    def _repeat_along_time(self, tensor, sequence_length):
        if tensor.ndim == 4:
            return tensor.unsqueeze(1).repeat(1, sequence_length, 1, 1, 1)
        if tensor.ndim == 3:
            return tensor.unsqueeze(1).repeat(1, sequence_length, 1, 1)
        raise ValueError(
            f"Cannot repeat tensor with shape {tuple(tensor.shape)} across time"
        )

    def _load_motion_sequence(self, data, prefix):
        key = prefix
        if key not in data:
            raise ValueError(f"MAT motion expects `{key}` in the dataloader output")

        tensor = data[key].to(self.device)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim != 5:
            raise ValueError(
                f"MAT motion expects `{key}` to be 4D or 5D, got shape {tuple(tensor.shape)}"
            )

        if self.opt.data_online_context_pixels > 0:
            tensor = tensor[
                :,
                :,
                :,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
            ]

        return tensor

    def _get_generator_inputs_for_current_batch(self):
        if self.opt.alg_mat_motion:
            return self.real_A_seq, self.mask_keep_seq, self.mask_class_seq
        return self.real_A, self.mask_keep, self.mask_class

    def set_input(self, data):
        super().set_input(data)

        use_ref_masks = not self.opt.alg_mat_motion
        self.input_A_label_mask = self._load_label_mask(
            data, "A", prefer_ref=use_ref_masks
        )
        self.input_B_label_mask = self._load_label_mask(
            data, "B", prefer_ref=use_ref_masks
        )

        label_mask = self.input_B_label_mask
        if label_mask is None:
            label_mask = self.input_A_label_mask
        if label_mask is None:
            raise ValueError(
                "MAT requires A_label_mask or B_label_mask in the dataloader output"
            )

        if self.opt.alg_mat_motion:
            source_a_seq = self._load_motion_sequence(data, "A")
            source_b_seq = self._load_motion_sequence(data, "B")
            sequence_length = source_a_seq.shape[1]
            if source_b_seq.shape[1] != sequence_length:
                raise ValueError(
                    "MAT motion expects A and B to have the same number of frames, "
                    f"got {source_a_seq.shape[1]} and {source_b_seq.shape[1]}"
                )
            if sequence_length < 2:
                raise ValueError(
                    "MAT motion expects at least 2 frames in the loaded window, "
                    f"got {sequence_length}"
                )
            if sequence_length != self._motion_sequence_length():
                raise ValueError(
                    "MAT motion dataloader returned an unexpected number of frames: "
                    f"expected {self._motion_sequence_length()}, got {sequence_length}"
                )

            if label_mask.ndim == 4:
                label_mask_seq = label_mask
            elif label_mask.ndim == 3:
                label_mask_seq = self._repeat_along_time(label_mask, sequence_length)
            else:
                raise ValueError(
                    "MAT motion expects label masks to be 3D or 4D after squeeze, "
                    f"got {tuple(label_mask.shape)}"
                )

            self.real_B_seq = source_b_seq
            self.real_A_seq = torch.cat(
                [source_b_seq[:, :-1], source_a_seq[:, -1:]], dim=1
            )

            hole_mask = (label_mask_seq > 0).to(dtype=self.real_A_seq.dtype).unsqueeze(2)
            self.mask_seq = torch.zeros_like(hole_mask)
            self.mask_seq[:, -1] = hole_mask[:, -1]
            self.mask_keep_seq = torch.ones_like(self.mask_seq)
            self.mask_keep_seq[:, -1] = 1.0 - self.mask_seq[:, -1]
            self.mask = self.mask_seq[:, -1]
            self.mask_keep = self.mask_keep_seq[:, -1]
            self.previous_frame = self.real_B_seq[:, -2]

            if self.opt.alg_mat_mask_class_conditioning:
                class_mask_last = (
                    label_mask_seq.to(dtype=self.real_A_seq.dtype).unsqueeze(2)
                    * hole_mask
                )
                self.mask_class_seq = torch.zeros_like(class_mask_last)
                self.mask_class_seq[:, -1] = class_mask_last[:, -1]
                self.mask_class = self.mask_class_seq[:, -1]
            else:
                self.mask_class_seq = None
                self.mask_class = None

            self.real_A = self.real_A_seq[:, -1]
            self.real_B = self.real_B_seq[:, -1]
        else:
            hole_mask = (label_mask > 0).to(dtype=self.real_A.dtype).unsqueeze(1)
            self.mask = hole_mask
            self.mask_keep = 1.0 - hole_mask
            self.mask_seq = None
            self.mask_keep_seq = None
            self.previous_frame = None
            self.real_A_seq = None
            self.real_B_seq = None
            if self.opt.alg_mat_mask_class_conditioning:
                class_mask = (
                    label_mask.to(dtype=self.real_A.dtype).unsqueeze(1) * hole_mask
                )
                self.mask_class = class_mask
            else:
                self.mask_class = None
            self.mask_class_seq = None

        self.y_t = self.real_A
        self.gt_image = self.real_B

    def _load_label_mask(self, data, prefix, prefer_ref=True):
        key = f"{prefix}_label_mask"
        if key not in data:
            return None

        label_mask = data[key].to(self.device)
        ref_key = f"{prefix}_ref_label_mask"
        if prefer_ref and ref_key in data:
            label_mask = data[ref_key].to(self.device)

        if label_mask.ndim == 5 and label_mask.shape[2] == 1:
            label_mask = label_mask.squeeze(2)
        elif label_mask.ndim == 4 and label_mask.shape[1] == 1:
            label_mask = label_mask.squeeze(1)
        elif label_mask.ndim not in {3, 4}:
            raise ValueError(
                f"Unsupported MAT label mask shape for {prefix}: {tuple(label_mask.shape)}"
            )

        if self.opt.data_online_context_pixels > 0:
            label_mask = label_mask[
                :,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
                self.opt.data_online_context_pixels : -self.opt.data_online_context_pixels,
            ]

        return label_mask

    def _unwrap_net(self, net):
        return net.module if hasattr(net, "module") else net

    def _get_generator_for_inference(self):
        if (
            not self.opt.isTrain
            and getattr(self.opt, "train_G_ema", False)
            and hasattr(self, "netG_A_ema")
        ):
            return self.netG_A_ema
        return self.netG_A

    def _sample_train_latent(self, batch_size):
        return torch.randn(
            batch_size,
            self.opt.alg_mat_z_dim,
            device=self.device,
            dtype=torch.float32,
        )

    def _normalize_image_paths_for_batch(self, batch_size, offset=0):
        paths = getattr(self, "image_paths", [])
        if isinstance(paths, str):
            paths = [paths]
        elif isinstance(paths, (list, tuple)):
            paths = list(paths)
        else:
            paths = []

        normalized = []
        for i in range(batch_size):
            if i < len(paths):
                normalized.append(str(paths[i]))
            else:
                normalized.append(f"batch_item_{offset + i}")
        return normalized

    def _stable_seed_from_path(self, path):
        digest = hashlib.sha256(path.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    def _sample_eval_latent(self, seed):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randn(
            self.opt.alg_mat_z_dim,
            generator=generator,
            dtype=torch.float32,
        ).to(self.device)

    def _map_with_style_mixing(self, netG, z, truncation_psi):
        ws = netG.mapping(z, None, truncation_psi=truncation_psi)
        if self.opt.alg_mat_style_mixing_prob <= 0 or ws.shape[1] <= 1:
            return ws

        cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(
            1, ws.shape[1]
        )
        should_mix = (
            torch.rand([], device=ws.device) < self.opt.alg_mat_style_mixing_prob
        )
        if should_mix:
            mixed_ws = netG.mapping(
                torch.randn_like(z),
                None,
                truncation_psi=truncation_psi,
                skip_w_avg_update=True,
            )
            ws[:, cutoff:] = mixed_ws[:, cutoff:]
        return ws

    def _forward_generator_train(self):
        netG = self._unwrap_net(self.netG_A)
        z = self._sample_train_latent(self.get_current_batch_size())
        ws = self._map_with_style_mixing(netG, z, truncation_psi=1.0)
        images_in, mask_keep, mask_class = self._get_generator_inputs_for_current_batch()
        self.fake_B, self.fake_B_stg1 = netG.synthesis(
            images_in,
            mask_keep,
            ws,
            mask_class=mask_class,
            noise_mode=self.opt.alg_mat_noise_mode_train,
            return_stg1=True,
        )
        self.fake_B = self._ensure_rgb_output(self.fake_B, "fake_B")
        self.fake_B_stg1 = self._ensure_rgb_output(self.fake_B_stg1, "fake_B_stg1")
        return self.fake_B, self.fake_B_stg1

    def forward(self):
        if self.opt.isTrain:
            self._forward_generator_train()
        else:
            images_in, mask_keep, mask_class = self._get_generator_inputs_for_current_batch()
            self.fake_B, self.fake_B_stg1 = self._run_eval_batch(
                images_in,
                mask_keep,
                offset=0,
                mask_class=mask_class,
            )

    def _run_eval_batch(self, images, mask_keep, offset=0, mask_class=None):
        if self.opt.alg_mat_motion and images.ndim == 4:
            sequence_length = self._motion_sequence_length()
            images = self._repeat_along_time(images, sequence_length)
            mask_keep = self._repeat_along_time(mask_keep, sequence_length)
            if mask_class is not None:
                mask_class = self._repeat_along_time(mask_class, sequence_length)

        netG = self._unwrap_net(self._get_generator_for_inference())
        paths = self._normalize_image_paths_for_batch(images.shape[0], offset=offset)
        was_training = netG.training
        netG.eval()

        outputs = []
        outputs_stg1 = []
        self.eval_seeds = []

        try:
            for i, path in enumerate(paths):
                seed = self._stable_seed_from_path(path)
                self.eval_seeds.append(seed)
                z = self._sample_eval_latent(seed).unsqueeze(0)
                device_ids = []
                if self.device.type == "cuda" and self.device.index is not None:
                    device_ids = [self.device.index]
                with torch.random.fork_rng(devices=device_ids):
                    torch.manual_seed(seed)
                    if self.device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    fake_B, fake_B_stg1 = netG(
                        images[i : i + 1],
                        mask_keep[i : i + 1],
                        z,
                        None,
                        mask_class=(
                            None
                            if mask_class is None
                            else mask_class[i : i + 1]
                        ),
                        truncation_psi=self.opt.alg_mat_truncation_psi,
                        noise_mode=self.opt.alg_mat_noise_mode_eval,
                        return_stg1=True,
                    )
                outputs.append(fake_B)
                outputs_stg1.append(fake_B_stg1)
        finally:
            netG.train(was_training)

        fake_B = self._ensure_rgb_output(torch.cat(outputs, dim=0), "fake_B")
        fake_B_stg1 = self._ensure_rgb_output(
            torch.cat(outputs_stg1, dim=0), "fake_B_stg1"
        )
        return fake_B, fake_B_stg1

    def _run_discriminator(self, images, images_stg1):
        netD = self._unwrap_net(self.netD_A)
        return netD(images, self.mask_keep, images_stg1, None)

    def compute_G_loss(self):
        gen_logits, gen_logits_stg1 = self._run_discriminator(
            self.fake_B, self.fake_B_stg1
        )
        pcp_loss, _ = self.criterionMAT(self.fake_B, self.real_B)

        self.loss_G_adv = F.softplus(-gen_logits).mean()
        self.loss_G_pcp = pcp_loss * self.opt.alg_mat_pcp_ratio
        self.loss_G_l1 = torch.mean(torch.abs(self.fake_B - self.real_B))
        if self.opt.alg_mat_motion:
            self.loss_G_adv_stg1 = torch.zeros(size=(), device=self.device)
            self.loss_G_tot = self.loss_G_adv + self.loss_G_pcp
        else:
            self.loss_G_adv_stg1 = F.softplus(-gen_logits_stg1).mean()
            self.loss_G_tot = self.loss_G_adv + self.loss_G_adv_stg1 + self.loss_G_pcp

    def compute_D_main_loss(self):
        fake_B = self.fake_B.detach()
        fake_B_stg1 = self.fake_B_stg1.detach()

        gen_logits, gen_logits_stg1 = self._run_discriminator(fake_B, fake_B_stg1)
        self.loss_D_fake = F.softplus(gen_logits).mean()

        real_B = self.real_B.detach()
        real_B_stg1 = self.real_B.detach()
        real_logits, real_logits_stg1 = self._run_discriminator(real_B, real_B_stg1)
        self.loss_D_real = F.softplus(-real_logits).mean()
        if self.opt.alg_mat_motion:
            self.loss_D_fake_stg1 = torch.zeros(size=(), device=self.device)
            self.loss_D_real_stg1 = torch.zeros(size=(), device=self.device)
            self.loss_D_tot = self.loss_D_fake + self.loss_D_real
        else:
            self.loss_D_fake_stg1 = F.softplus(gen_logits_stg1).mean()
            self.loss_D_real_stg1 = F.softplus(-real_logits_stg1).mean()
            self.loss_D_tot = (
                self.loss_D_fake
                + self.loss_D_fake_stg1
                + self.loss_D_real
                + self.loss_D_real_stg1
            )

    def compute_D_r1_loss(self):
        real_B = self.real_B.detach().requires_grad_(True)
        real_B_stg1 = self.real_B.detach().requires_grad_(True)
        real_logits, real_logits_stg1 = self._run_discriminator(real_B, real_B_stg1)

        self.loss_D_r1 = torch.zeros(size=(), device=self.device)
        self.loss_D_r1_stg1 = torch.zeros(size=(), device=self.device)
        with conv2d_gradfix.no_weight_gradients():
            r1_grads = torch.autograd.grad(
                outputs=[real_logits.sum()],
                inputs=[real_B],
                create_graph=True,
                only_inputs=True,
            )[0]
            if not self.opt.alg_mat_motion:
                r1_grads_stg1 = torch.autograd.grad(
                    outputs=[real_logits_stg1.sum()],
                    inputs=[real_B_stg1],
                    create_graph=True,
                    only_inputs=True,
                )[0]

        r1_penalty = r1_grads.square().sum([1, 2, 3]).mean()
        self.loss_D_r1 = r1_penalty * (self.opt.alg_mat_r1_gamma / 2)
        if self.opt.alg_mat_motion:
            self.loss_D_r1_stg1 = torch.zeros(size=(), device=self.device)
        else:
            r1_penalty_stg1 = r1_grads_stg1.square().sum([1, 2, 3]).mean()
            self.loss_D_r1_stg1 = r1_penalty_stg1 * (self.opt.alg_mat_r1_gamma / 2)

    def _scaled_backward(self, loss):
        scaled_loss = loss / self.opt.train_iter_size
        if self.use_cuda:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def _sanitize_optimizer_grads(self, optimizer_names):
        if self.niter % self.opt.train_iter_size != 0:
            return

        optimizers = [
            getattr(self, optimizer_name) for optimizer_name in optimizer_names
        ]
        if self.use_cuda:
            for optimizer in optimizers:
                self.scaler.unscale_(optimizer)

        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        torch.nan_to_num(
                            param.grad,
                            nan=0.0,
                            posinf=1e5,
                            neginf=-1e5,
                            out=param.grad,
                        )

    def _update_mat_ema(self):
        if not self.opt.train_G_ema or self.niter % self.opt.train_iter_size != 0:
            return

        netG = self._unwrap_net(self.netG_A)
        netG_ema = getattr(self, "netG_A_ema", None)
        if netG_ema is None:
            self.netG_A_ema = copy.deepcopy(netG).eval()
            netG_ema = self.netG_A_ema

        batch_size = self.get_current_batch_size() * max(1, len(self.opt.gpu_ids))
        batch_size *= self.opt.train_iter_size
        ema_nimg = self.opt.alg_mat_ema_kimg * 1000.0
        if self.opt.alg_mat_ema_rampup > 0:
            current_nimg = (
                self.niter
                * self.get_current_batch_size()
                * max(1, len(self.opt.gpu_ids))
            )
            ema_nimg = min(ema_nimg, current_nimg * self.opt.alg_mat_ema_rampup)
        ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))

        with torch.no_grad():
            for p_ema, p in zip(netG_ema.parameters(), netG.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(netG_ema.buffers(), netG.buffers()):
                b_ema.copy_(b)

    def _adapt_loaded_conv_weight(self, model_weight, loaded_weight):
        if model_weight.ndim != 4 or loaded_weight.ndim != 4:
            return None
        if model_weight.shape[0] != loaded_weight.shape[0]:
            return None
        if model_weight.shape[2:] != loaded_weight.shape[2:]:
            return None
        if model_weight.shape[1] <= loaded_weight.shape[1]:
            return None

        expanded = model_weight.detach().clone()
        expanded.zero_()
        expanded[:, : loaded_weight.shape[1]] = loaded_weight.to(expanded)
        return expanded

    def _load_generator_state_dict_for_motion(self, net, state_dict, allow_partial=False):
        model_dict = net.state_dict()
        filtered = {}
        unexpected = []
        mismatched = []
        adapted = []

        for key, value in state_dict.items():
            if key not in model_dict:
                unexpected.append(key)
                continue
            if model_dict[key].shape != value.shape:
                adapted_value = self._adapt_loaded_conv_weight(model_dict[key], value)
                if adapted_value is not None:
                    filtered[key] = adapted_value
                    adapted.append((key, tuple(value.shape), tuple(model_dict[key].shape)))
                    continue
                mismatched.append((key, tuple(value.shape), tuple(model_dict[key].shape)))
                continue
            filtered[key] = value

        missing = [key for key in model_dict.keys() if key not in filtered]
        allowed_missing = set()
        if self.opt.alg_mat_motion:
            allowed_missing.update(
                key for key in missing if key.startswith("synthesis.motion_module.")
            )
        disallowed_missing = [key for key in missing if key not in allowed_missing]

        if adapted:
            adapted_preview = ", ".join(
                f"{key} {src_shape}->{dst_shape}"
                for key, src_shape, dst_shape in adapted[:10]
            )
            print("MAT fine-tune adapted widened input weights:", adapted_preview)

        if allow_partial:
            if unexpected:
                print(
                    "Skipping unexpected MAT generator keys:",
                    ", ".join(sorted(unexpected)[:10]),
                )
            if mismatched:
                print(
                    "Skipping unmatched MAT generator keys:",
                    ", ".join(key for key, _, _ in mismatched[:10]),
                )
            if disallowed_missing:
                print(
                    "Leaving MAT generator keys randomly initialized:",
                    ", ".join(sorted(disallowed_missing)[:10]),
                )
        elif unexpected or mismatched or disallowed_missing:
            details = []
            if unexpected:
                details.append(
                    "unexpected keys: " + ", ".join(sorted(unexpected)[:10])
                )
            if mismatched:
                mismatch_preview = ", ".join(
                    f"{key} {src_shape}->{dst_shape}"
                    for key, src_shape, dst_shape in mismatched[:10]
                )
                details.append("shape mismatches: " + mismatch_preview)
            if disallowed_missing:
                details.append(
                    "missing non-motion keys: "
                    + ", ".join(sorted(disallowed_missing)[:10])
                )
            raise RuntimeError(
                "MAT fine-tuning checkpoint loading only tolerates missing motion-module "
                "generator keys and widened input convs for mask-class conditioning. "
                + " | ".join(details)
            )

        net.load_state_dict(filtered, strict=False)

    def load_networks(self, epoch):
        for name in self.model_names:
            if not isinstance(name, str):
                continue

            load_filename = f"{epoch}_net_{name}.pth"
            load_path = os.path.join(self.save_dir, load_filename)
            ema_load_filename = f"{epoch}_net_{name}_ema.pth"
            ema_load_path = os.path.join(self.save_dir, ema_load_filename)

            load_path_effective = load_path
            if (
                not self.opt.isTrain
                and getattr(self.opt, "train_G_ema", False)
                and os.path.isfile(ema_load_path)
            ):
                load_path_effective = ema_load_path

            net = getattr(self, "net" + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module

            print(f"loading the model from {load_path_effective}")
            state_dict = torch.load(load_path_effective, map_location=str(self.device))
            if hasattr(state_dict, "_metadata"):
                del state_dict._metadata
            if isinstance(state_dict, dict) and "_ema" in state_dict:
                state_dict = state_dict["_ema"]

            if name == "G_A" and (
                self.opt.alg_mat_motion or self.opt.alg_mat_mask_class_conditioning
            ):
                self._load_generator_state_dict_for_motion(
                    net,
                    state_dict,
                    allow_partial=self.opt.model_load_no_strictness,
                )
            else:
                net.load_state_dict(
                    state_dict, strict=self.opt.model_load_no_strictness
                )

    def optimize_parameters(self):
        self.niter += 1

        autocast_enabled = self.with_amp and self.use_cuda
        with ExitStack() as stack:
            if len(self.opt.gpu_ids) > 1 and self.niter % self.opt.train_iter_size != 0:
                stack.enter_context(self.netG_A.no_sync())
                stack.enter_context(self.netD_A.no_sync())

            self.set_requires_grad(self.netD_A, False)
            self._set_generator_trainable(True)
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                self._forward_generator_train()
                self.compute_G_loss()
            self._scaled_backward(self.loss_G_tot)
            self._sanitize_optimizer_grads(["optimizer_G"])
            self.compute_step(["optimizer_G"], self.loss_names_G)

            self.set_requires_grad(self.netD_A, True)
            self._set_generator_trainable(False)

            with torch.no_grad():
                self._forward_generator_train()
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                self.compute_D_main_loss()
                self.loss_D_r1 = torch.zeros(size=(), device=self.device)
                self.loss_D_r1_stg1 = torch.zeros(size=(), device=self.device)
            self._scaled_backward(self.loss_D_tot)
            self._sanitize_optimizer_grads(["optimizer_D"])

            apply_r1 = self.opt.alg_mat_d_reg_every > 0 and (
                self.niter % self.opt.alg_mat_d_reg_every == 0
            )
            if apply_r1:
                self.compute_step(
                    ["optimizer_D"],
                    ["D_fake", "D_fake_stg1", "D_real", "D_real_stg1"],
                )

                with torch.amp.autocast("cuda", enabled=autocast_enabled):
                    self.compute_D_r1_loss()
                    if self.opt.alg_mat_motion:
                        self.loss_D_tot = (
                            self.loss_D_fake
                            + self.loss_D_real
                            + self.loss_D_r1
                        )
                    else:
                        self.loss_D_tot = (
                            self.loss_D_fake
                            + self.loss_D_fake_stg1
                            + self.loss_D_real
                            + self.loss_D_real_stg1
                            + self.loss_D_r1
                            + self.loss_D_r1_stg1
                        )
                self._scaled_backward(
                    (self.loss_D_r1 + self.loss_D_r1_stg1)
                    * self.opt.alg_mat_d_reg_every
                )
                self._sanitize_optimizer_grads(["optimizer_D"])
                self.compute_step(["optimizer_D"], ["D_r1", "D_r1_stg1", "D_tot"])
            else:
                self.compute_step(["optimizer_D"], self.loss_names_D)

        self._update_mat_ema()

    def inference(self, nb_imgs, offset=0):
        images_in, mask_keep, mask_class = self._get_generator_inputs_for_current_batch()
        with torch.no_grad():
            fake_B, fake_B_stg1 = self._run_eval_batch(
                images_in[:nb_imgs],
                mask_keep[:nb_imgs],
                offset=offset,
                mask_class=None if mask_class is None else mask_class[:nb_imgs],
            )

        self.fake_B = fake_B
        self.fake_B_stg1 = fake_B_stg1
        self.output = fake_B

        for name in self.gen_visual_names:
            whole_tensor = getattr(self, name[:-1])
            for k in range(min(nb_imgs, self.get_current_batch_size())):
                cur_tensor = whole_tensor[k : k + 1]
                if "mask" in name:
                    cur_tensor = cur_tensor.squeeze(0)
                setattr(self, name + str(offset + k), cur_tensor)

    def compute_visuals(self, nb_imgs):
        super().compute_visuals(nb_imgs)
        self.inference(nb_imgs)
