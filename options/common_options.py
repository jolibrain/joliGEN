import os
import argparse

import torch

from .base_options import BaseOptions
from util.util import MAX_INT, pairs_of_ints, pairs_of_floats
from models.modules.classifiers import TORCH_MODEL_CLASSES
from models import get_models_names, get_option_setter
import models
import data


class CommonOptions(BaseOptions):
    def __init__(self):
        super().__init__()

        self.opt_schema = {
            "properties": {
                "D": {"title": "Discriminator"},
                "G": {"title": "Generator"},
                "alg": {
                    "title": "Algorithm-specific",
                    "properties": {
                        "gan": {"title": "GAN model"},
                        "cut": {"title": "CUT model"},
                        "cyclegan": {"title": "CycleGAN model"},
                        "re": {"title": "ReCUT / ReCycleGAN"},
                        "palette": {"title": "Diffusion model"},
                    },
                },
                "data": {
                    "title": "Datasets",
                    "properties": {
                        "online_creation": {"title": "Online created datasets"}
                    },
                },
                "f_s": {"title": "Semantic segmentation network"},
                "cls": {"title": "Semantic classification network"},
                "output": {
                    "title": "Output",
                    "properties": {"display": {"title": "Visdom display"}},
                },
                "model": {"title": "Model"},
                "train": {
                    "title": "Training",
                    "properties": {
                        "sem": {"title": "Semantic training"},
                        "mask": {"title": "Semantic training with masks"},
                    },
                },
                "dataaug": {"title": "Data augmentation"},
            }
        }

        # Options that should stay at the root of the schema
        self.general_options = ["model_type"]

    def initialize(self, parser):
        super().initialize(parser)

        # basic parameters
        parser.add_argument(
            "--dataroot",
            type=str,
            required=True,
            help="path to images (should have subfolders trainA, trainB, valA, valB, etc)",
        )
        parser.add_argument(
            "--name",
            type=str,
            default="experiment_name",
            help="name of the experiment. It decides where to store samples and models",
        )
        parser.add_argument(
            "--suffix",
            default="",
            type=str,
            help="customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}",
        )
        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
        )
        parser.add_argument(
            "--with_amp",
            action="store_true",
            help="whether to activate torch amp on forward passes",
        )
        parser.add_argument(
            "--with_tf32",
            action="store_true",
            help="whether to activate tf32 for faster computations (Ampere GPU and beyond only)",
        )
        parser.add_argument(
            "--with_torch_compile",
            action="store_true",
            help="whether to activate torch.compile for some forward and backward functions (experimental)",
        )
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./checkpoints",
            help="models are saved here",
        )
        parser.add_argument(
            "--phase", type=str, default="train", help="train, val, test, etc"
        )
        parser.add_argument("--ddp_port", type=str, default="12355")
        parser.add_argument(
            "--warning_mode", action="store_true", help="whether to display warning"
        )

        # model parameters
        parser.add_argument(
            "--model_type",
            type=str,
            default="cut",
            choices=["cut", "cycle_gan", "palette", "cm", "cm_gan", "sc"],
            help="chooses which model to use.",
        )
        parser.add_argument(
            "--model_input_nc",
            type=int,
            default=3,
            help="# of input image channels: 3 for RGB and 1 for grayscale, more supported",
        )
        parser.add_argument(
            "--model_output_nc",
            type=int,
            default=3,
            # choices=[1, 3],
            help="# of output image channels: 3 for RGB and 1 for grayscale",
        )
        parser.add_argument(
            "--model_init_type",
            type=str,
            default="normal",
            choices=["normal", "xavier", "kaiming", "orthogonal"],
            help="network initialization",
        )
        parser.add_argument(
            "--model_init_gain",
            type=float,
            default=0.02,
            help="scaling factor for normal, xavier and orthogonal.",
        )
        parser.add_argument(
            "--model_multimodal",
            action="store_true",
            help="multimodal model with random latent input vector",
        )
        parser.add_argument(
            "--model_load_no_strictness",
            action="store_true",
            help="load model without strictness check (strict=False)",
        )

        # depth network
        parser.add_argument(
            "--model_depth_network",
            type=str,
            default="DPT_Large",
            choices=[
                "DPT_Large",
                "DPT_Hybrid",  # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
                "MiDaS_small",
                "DPT_BEiT_L_512",
                "DPT_BEiT_L_384",
                "DPT_BEiT_B_384",
                "DPT_SwinV2_L_384",
                "DPT_SwinV2_B_384",
                "DPT_SwinV2_T_256",
                "DPT_Swin_L_384",
                "DPT_Next_ViT_L_384",
                "DPT_LeViT_224",
            ],  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
            help="specify depth prediction network architecture",
        )

        parser.add_argument(
            "--D_weight_sam",
            type=str,
            default="",
            help="path to sam weight for D, e.g. models/configs/sam/pretrain/sam_vit_b_01ec64.pth, or models/configs/sam/pretrain/mobile_sam.pt for MobileSAM",
        )

        # generator
        parser.add_argument(
            "--G_ngf",
            type=int,
            default=64,
            help="# of gen filters in the last conv layer",
        )
        parser.add_argument(
            "--G_netG",
            type=str,
            default="mobile_resnet_attn",
            choices=[
                "resnet",
                "resnet_attn",
                "mobile_resnet",
                "mobile_resnet_attn",
                "unet_256",
                "unet_128",
                "segformer_attn_conv",
                "segformer_conv",
                "ittr",
                "unet_mha",
                "uvit",
                "unet_mha_ref_attn",
                "dit",
                "hdit",
                "img2img_turbo",
                "unet_vid",
                "hat",
                "vit",
            ],
            help="specify generator architecture",
        )
        parser.add_argument(
            "--G_nblocks",
            type=int,
            default=9,
            help="# of layer blocks in G, applicable to resnets",
        )
        parser.add_argument(
            "--G_dropout", action="store_true", help="dropout for the generator"
        )
        parser.add_argument(
            "--G_spectral",
            action="store_true",
            help="whether to use spectral norm in the generator",
        )
        parser.add_argument(
            "--G_padding_type",
            type=str,
            choices=["reflect", "replicate", "zeros"],
            help="whether to use padding in the generator",
            default="reflect",
        )
        parser.add_argument(
            "--G_norm",
            type=str,
            default="instance",
            choices=["instance", "batch", "none"],
            help="instance normalization or batch normalization for G",
        )
        parser.add_argument(
            "--G_config_segformer",
            type=str,
            default="models/configs/segformer/segformer_config_b0.json",
            help="path to segformer configuration file for G",
        )
        parser.add_argument(
            "--G_attn_nb_mask_attn",
            default=10,
            type=int,
            help="number of attention masks in _attn model architectures",
        )

        parser.add_argument(
            "--G_attn_nb_mask_input",
            default=1,
            type=int,
            help="number of mask dedicated to input in _attn model architectures",
        )

        parser.add_argument(
            "--G_backward_compatibility_twice_resnet_blocks",
            action="store_true",
            help="if true, feats will go througt resnet blocks two times for resnet_attn generators. This option will be deleted, it's for backward compatibility (old models were trained that way).",
        )
        parser.add_argument(
            "--G_netE",
            type=str,
            default="resnet_256",
            choices=[
                "resnet_128",
                "resnet_256",
                "resnet_512",
                "conv_128",
                "conv_256",
                "conv_512",
            ],
            help="specify multimodal latent vector encoder",
        )

        parser.add_argument(
            "--G_unet_mha_num_head_channels",
            default=32,
            type=int,
            help="number of channels in each head of the mha architecture",
        )
        parser.add_argument(
            "--G_unet_mha_num_heads",
            default=1,
            type=int,
            help="number of heads in the mha architecture",
        )

        parser.add_argument(
            "--G_uvit_num_transformer_blocks",
            default=6,
            type=int,
            help="Number of transformer blocks in UViT",
        )

        parser.add_argument(
            "--G_diff_n_timestep_train",
            type=int,
            default=2000,
            help="Number of timesteps used for UNET mha training.",
        )

        parser.add_argument(
            "--G_diff_n_timestep_test",
            type=int,
            default=1000,
            help="Number of timesteps used for UNET mha inference (test time).",
        )

        parser.add_argument(
            "--G_unet_mha_res_blocks",
            default=[2, 2, 2, 2],
            nargs="*",
            type=int,
            help="distribution of resnet blocks across the UNet stages, should have same size as --G_unet_mha_channel_mults",
        )

        parser.add_argument(
            "--G_unet_mha_channel_mults",
            default=[1, 2, 4, 8],
            nargs="*",
            type=int,
            help="channel multiplier for each level of the UNET mha",
        )

        parser.add_argument(
            "--G_unet_mha_attn_res",
            default=[16],
            nargs="*",
            type=int,
            help="downrate samples at which attention takes place",
        )

        parser.add_argument(
            "--G_unet_mha_norm_layer",
            type=str,
            choices=[
                "groupnorm",
                "batchnorm",
                "layernorm",
                "instancenorm",
                "switchablenorm",
            ],
            default="groupnorm",
        )

        parser.add_argument(
            "--G_unet_mha_group_norm_size",
            type=int,
            default=32,
        )

        parser.add_argument(
            "--G_unet_mha_vit_efficient",
            action="store_true",
            help="if true, use efficient attention in UNet and UViT",
        )

        parser.add_argument(
            "--G_hdit_depths",
            default=[2, 2, 4],
            nargs="*",
            type=int,
            help="distribution of depth blocks across the HDiT stages, should have same size as --G_hdit_widths",
        )

        parser.add_argument(
            "--G_hdit_widths",
            default=[192, 384, 768],
            nargs="*",
            type=int,
            help="width multiplier for each level of the HDiT",
        )

        parser.add_argument(
            "--G_hdit_patch_size",
            default=4,
            type=int,
            help="Patch size for HDIT, e.g. 4 for 4x4 patches",
        )
        parser.add_argument(
            "--G_unet_vid_max_sequence_length",
            default=25,
            type=int,
            help="max frame number(sequence length) for unet_vid in the PositionalEncoding",
        )
        parser.add_argument(
            "--G_unet_vid_num_attention_heads",
            default=8,
            type=int,
            help="number of attention heads for unet_vid motion module, 8, 4, ...",
        )
        parser.add_argument(
            "--G_unet_vid_num_transformer_blocks",
            default=2,
            type=int,
            help="number of unet_vid motion module transformer blocks, 2, 1, ...",
        )
        parser.add_argument(
            "--G_lora_unet",
            type=int,
            default=8,
            help="lora unet rank for G",
        )
        parser.add_argument(
            "--G_lora_vae",
            type=int,
            default=8,
            help="lora vae rank for G",
        )
        parser.add_argument(
            "--G_vit_variant",
            type=str,
            default="JiT-B/16",
            help="Selects the ViT backbone when --G_netG vit",
        )
        parser.add_argument(
            "--G_vit_depth", type=int, default=12, help="JiT transformer depth"
        )

        parser.add_argument(
            "--G_vit_num_classes",
            type=int,
            default=1,
            help="Number of class embeddings for vit",
        )

        parser.add_argument(
            "--G_vit_bottleneck_dim",
            type=int,
            default=128,
            help="Numbe embeddings for vit",
        )
        parser.add_argument(
            "--G_vit_hidden_size",
            type=int,
            default=728,
            help="Numbe embeddings for vit",
        )
        parser.add_argument(
            "--G_vit_num_heads",
            type=int,
            default=12,
            help="Numbe heads for vit",
        )
        parser.add_argument(
            "--G_vit_patch_size",
            type=int,
            default=16,
            help="Numbe patch for vit",
        )

        # discriminator
        parser.add_argument(
            "--D_ndf",
            type=int,
            default=64,
            help="# of discrim filters in the first conv layer",
        )
        parser.add_argument(
            "--D_netDs",
            type=str,
            default=["projected_d", "basic"],
            choices=[
                "basic",
                "n_layers",
                "pixel",
                "projected_d",
                "temporal",
                "vision_aided",
                "depth",
                "mask",
                "sam",
                "none",
            ]
            + list(TORCH_MODEL_CLASSES.keys()),
            help="specify discriminator architecture, another option, --D_n_layers allows you to specify the layers in the n_layers discriminator. NB: duplicated arguments are ignored. Values: basic, n_layers, pixel, projected_d, temporal, vision_aided, depth, mask, sam",
            nargs="+",
        )
        parser.add_argument(
            "--D_vision_aided_backbones",
            type=str,
            default="clip+dino+swin",
            help="specify vision aided discriminators architectures, they are frozen then output are combined and fitted with a linear network on top, choose from dino, clip, swin, det_coco, seg_ade and combine them with +",
        )
        parser.add_argument(
            "--D_n_layers", type=int, default=3, help="only used if netD==n_layers"
        )
        parser.add_argument(
            "--D_norm",
            type=str,
            default="instance",
            choices=["instance", "batch", "none"],
            help="instance normalization or batch normalization for D",
        )
        parser.add_argument(
            "--D_dropout",
            action="store_true",
            help="whether to use dropout in the discriminator",
        )
        parser.add_argument(
            "--D_spectral",
            action="store_true",
            help="whether to use spectral norm in the discriminator",
        )
        parser.add_argument(
            "--D_proj_interp",
            type=int,
            default=-1,
            help="whether to force projected discriminator interpolation to a value > 224, -1 means no interpolation",
        )
        parser.add_argument(
            "--D_proj_network_type",
            type=str,
            default="efficientnet",
            choices=[
                "efficientnet",
                "segformer",
                "vitbase",
                "vitsmall",
                "vitsmall2",
                "vitclip16",
                "vitclip14",
                "depth",
                "dinov2_vits14",
                "dinov2_vitb14",
                "dinov2_vitl14",
                "dinov2_vitg14",
                "dinov2_vits14_reg",
                "dinov2_vitb14_reg",
                "dinov2_vitl14_reg",
                "dinov2_vitg14_reg",
                "siglip_vitb16",
                "siglip_vitl16",
                "siglip_vit_so400m",
            ],
            help="projected discriminator architecture",
        )
        parser.add_argument(
            "--D_no_antialias",
            action="store_true",
            help="if specified, use stride=2 convs instead of antialiased-downsampling (sad)",
        )
        parser.add_argument(
            "--D_no_antialias_up",
            action="store_true",
            help="if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]",
        )
        parser.add_argument(
            "--D_proj_config_segformer",
            type=str,
            default="models/configs/segformer/segformer_config_b0.json",
            help="path to segformer configuration file",
        )
        parser.add_argument(
            "--D_proj_weight_segformer",
            type=str,
            default="models/configs/segformer/pretrain/segformer_mit-b0.pth",
            help="path to segformer weight",
        )

        parser.add_argument(
            "--D_temporal_every",
            type=int,
            default=4,
            help="apply temporal discriminator every x steps",
        )

        # mask semantic network : f_s
        parser.add_argument(
            "--f_s_net",
            type=str,
            default="vgg",
            choices=["vgg", "unet", "segformer", "sam"],
            help="specify f_s network [vgg|unet|segformer|sam]",
        )
        parser.add_argument(
            "--f_s_dropout",
            action="store_true",
            help="dropout for the semantic network",
        )
        parser.add_argument(
            "--f_s_semantic_nclasses",
            default=2,
            type=int,
            help="number of classes of the semantic loss classifier",
        )
        parser.add_argument(
            "--f_s_class_weights",
            default=[],
            nargs="*",
            type=int,
            help="class weights for imbalanced semantic classes",
        )
        parser.add_argument(
            "--f_s_semantic_threshold",
            default=1.0,
            type=float,
            help="threshold of the semantic classifier loss below with semantic loss is applied",
        )
        parser.add_argument(
            "--f_s_all_classes_as_one",
            action="store_true",
            help="if true, all classes will be considered as the same one (ie foreground vs background)",
        )
        parser.add_argument(
            "--f_s_nf",
            type=int,
            default=64,
            help="# of filters in the first conv layer of classifier",
        )
        parser.add_argument(
            "--f_s_config_segformer",
            type=str,
            default="models/configs/segformer/segformer_config_b0.json",
            help="path to segformer configuration file for f_s",
        )
        parser.add_argument(
            "--f_s_weight_segformer",
            type=str,
            default="",
            help="path to segformer weight for f_s, e.g. models/configs/segformer/pretrain/segformer_mit-b0.pth",
        )
        parser.add_argument(
            "--f_s_weight_sam",
            type=str,
            default="",
            help="path to sam weight for f_s, e.g. models/configs/sam/pretrain/sam_vit_b_01ec64.pth, or models/configs/sam/pretrain/mobile_sam.pt for MobileSAM",
        )

        # class semantic network : cls
        parser.add_argument(
            "--cls_net",
            type=str,
            default="vgg",
            choices=["vgg", "unet", "segformer"],
            help="specify cls network [vgg|unet|segformer]",
        )
        parser.add_argument(
            "--cls_dropout",
            action="store_true",
            help="dropout for the semantic network",
        )
        parser.add_argument(
            "--cls_semantic_nclasses",
            default=2,
            type=int,
            help="number of classes of the semantic loss classifier",
        )
        parser.add_argument(
            "--cls_class_weights",
            default=[],
            nargs="*",
            type=int,
            help="class weights for imbalanced semantic classes",
        )
        parser.add_argument(
            "--cls_semantic_threshold",
            default=1.0,
            type=float,
            help="threshold of the semantic classifier loss below with semantic loss is applied",
        )
        parser.add_argument(
            "--cls_all_classes_as_one",
            action="store_true",
            help="if true, all classes will be considered as the same one (ie foreground vs background)",
        )
        parser.add_argument(
            "--cls_nf",
            type=int,
            default=64,
            help="# of filters in the first conv layer of classifier",
        )
        parser.add_argument(
            "--cls_config_segformer",
            type=str,
            default="models/configs/segformer/segformer_config_b0.json",
            help="path to segformer configuration file for cls",
        )
        parser.add_argument(
            "--cls_weight_segformer",
            type=str,
            default="",
            help="path to segformer weight for cls, e.g. models/configs/segformer/pretrain/segformer_mit-b0.pth",
        )

        # dataset parameters
        parser.add_argument(
            "--data_dataset_mode",
            type=str,
            default="unaligned",
            choices=[
                "unaligned",
                "unaligned_labeled_cls",
                "unaligned_labeled_mask",
                "self_supervised_labeled_mask",
                "unaligned_labeled_mask_cls",
                "self_supervised_labeled_mask_cls",
                "unaligned_labeled_mask_online",
                "self_supervised_labeled_mask_online",
                "unaligned_labeled_mask_cls_online",
                "self_supervised_labeled_mask_cls_online",
                "aligned",
                "nuplet_unaligned_labeled_mask",
                "temporal_labeled_mask_online",
                "self_supervised_vid_mask_online",
                "self_supervised_vid_labeled_mask_cls_online",
                "self_supervised_temporal",
                "single",
                "unaligned_labeled_mask_ref",
                "self_supervised_labeled_mask_ref",
                "unaligned_labeled_mask_online_ref",
                "unaligned_labeled_mask_online_prompt",
                "self_supervised_labeled_mask_online_ref",
            ],
            help="chooses how datasets are loaded.",
        )
        parser.add_argument(
            "--data_direction",
            type=str,
            default="AtoB",
            choices=["AtoB", "BtoA"],
            help="AtoB or BtoA",
        )
        parser.add_argument(
            "--data_image_bits",
            type=int,
            default=8,
            help="number of bits of the image (e.g. 8, 12 or 16)",
        )
        parser.add_argument(
            "--data_serial_batches",
            action="store_true",
            help="if true, takes images in order to make batches, otherwise takes them randomly",
        )
        parser.add_argument(
            "--data_num_threads", default=4, type=int, help="# threads for loading data"
        )

        parser.add_argument(
            "--data_load_size", type=int, default=286, help="scale images to this size"
        )
        parser.add_argument(
            "--data_crop_size", type=int, default=256, help="then crop to this size"
        )
        parser.add_argument(
            "--data_max_dataset_size",
            type=int,
            default=MAX_INT,
            help="Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.",
        )
        parser.add_argument(
            "--data_preprocess",
            type=str,
            default="resize_and_crop",
            choices=[
                "resize_and_crop",
                "crop",
                "scale_width",
                "scale_width_and_crop",
                "none",
            ],
            help="scaling and cropping of images at load time",
        )

        parser.add_argument(
            "--data_refined_mask",
            action="store_true",
            help="whether to use refined mask with sam",
        )

        parser.add_argument(
            "--model_type_sam",
            type=str,
            default="mobile_sam",
            choices=["sam", "mobile_sam"],
            help="which model to use for segment-anything mask generation",
        )

        # Online dataset creation options
        parser.add_argument(
            "--data_online_select_category",
            default=-1,
            type=int,
            help="category to select for bounding boxes, -1 means all boxes selected",
        )
        parser.add_argument(
            "--data_online_single_bbox",
            action="store_true",
            help="whether to only allow a single bbox per online crop",
        )
        parser.add_argument(
            "--data_online_random_bbox",
            action="store_true",
            help="whether to randomly sample a bbox per online crop",
        )
        parser.add_argument(
            "--data_online_creation_load_size_A",
            default=[],
            nargs="*",
            type=int,
            help="load to this size during online creation, format : width height or only one size if square",
        )
        parser.add_argument(
            "--data_online_creation_crop_size_A",
            type=int,
            default=512,
            help="crop to this size during online creation, it needs to be greater than bbox size for domain A",
        )
        parser.add_argument(
            "--data_online_creation_crop_delta_A",
            type=int,
            default=50,
            help="size of crops are random, values allowed are online_creation_crop_size more or less online_creation_crop_delta for domain A",
        )
        parser.add_argument(
            "--data_online_creation_mask_delta_A",
            default=[[]],
            type=pairs_of_ints,
            nargs="+",
            help="mask offset (in pixels) to allow generation of a bigger object in domain B (for semantic loss) for domain A, format : 'width (x),height (y)' for each class or only one size if square, e.g. '125, 55 100, 100' for 2 classes",
        )

        parser.add_argument(
            "--data_online_creation_mask_delta_A_ratio",
            default=[[]],
            type=pairs_of_floats,
            nargs="+",
            help="ratio mask offset to allow generation of a bigger object in domain B (for semantic loss) for domain A, format : width (x),height (y) for each class or only one size if square",
        )

        parser.add_argument(
            "--data_online_creation_mask_random_offset_A",
            type=float,
            default=[0.0],
            nargs="*",
            help="ratio mask size randomization (only to make bigger one) to robustify the image generation in domain A, format : width (x) height (y) or only one size if square",
        )

        parser.add_argument(
            "--data_online_creation_mask_square_A",
            action="store_true",
            help="whether masks should be squared for domain A",
        )
        parser.add_argument(
            "--data_online_creation_rand_mask_A",
            action="store_true",
            help="Perform task of replacing noised masks by objects",
        )
        parser.add_argument(
            "--data_online_creation_color_mask_A",
            action="store_true",
            help="Perform task of replacing color-filled masks by objects",
        )

        parser.add_argument(
            "--data_online_creation_load_size_B",
            default=[],
            nargs="*",
            type=int,
            help="load to this size during online creation, format : width height or only one size if square",
        )

        parser.add_argument(
            "--data_online_creation_crop_size_B",
            type=int,
            default=512,
            help="crop to this size during online creation, it needs to be greater than bbox size for domain B",
        )
        parser.add_argument(
            "--data_online_creation_crop_delta_B",
            type=int,
            default=50,
            help="size of crops are random, values allowed are online_creation_crop_size more or less online_creation_crop_delta for domain B",
        )
        parser.add_argument(
            "--data_online_creation_mask_delta_B",
            default=[[]],
            type=pairs_of_ints,
            nargs="+",
            help="mask offset (in pixels) to allow generation of a bigger object in domain A (for semantic loss) for domain B, format : 'width (x),height (y)' for each class or only one size if square, e.g. '125, 55 100, 100' for 2 classes",
        )

        parser.add_argument(
            "--data_online_creation_mask_delta_B_ratio",
            default=[[]],
            type=pairs_of_floats,
            nargs="+",
            help="ratio mask offset to allow generation of a bigger object in domain A (for semantic loss) for domain B, format : 'width (x),height (y)' for each class or only one size if square",
        )

        parser.add_argument(
            "--data_online_creation_mask_random_offset_B",
            type=float,
            default=[0.0],
            nargs="*",
            help="mask size randomization (only to make bigger one) to robustify the image generation in domain B, format : width (y) height (x) or only one size if square",
        )

        parser.add_argument(
            "--data_online_creation_mask_square_B",
            action="store_true",
            help="whether masks should be squared for domain B",
        )
        parser.add_argument(
            "--data_online_context_pixels",
            type=int,
            default=0,
            help="context pixel band around the crop, unused for generation, only for disc ",
        )

        parser.add_argument(
            "--data_online_fixed_mask_size",
            type=int,
            default=-1,
            help="if >0, it will be used as fixed bbox size (warning: in dataset resolution ie before resizing) ",
        )

        # data temporal options
        parser.add_argument(
            "--data_temporal_number_frames",
            type=int,
            default=5,
            help="how many successive frames use for temporal loader",
        )

        parser.add_argument(
            "--data_temporal_frame_step",
            type=int,
            default=30,
            help="how many frames between successive frames selected",
        )

        parser.add_argument(
            "--data_temporal_num_common_char",
            type=int,
            default=-1,
            help="how many characters (the first ones) are used to identify a video; if =-1 natural sorting is used ",
        )

        # other data options
        parser.add_argument(
            "--data_inverted_mask",
            action="store_true",
            help="whether to invert the mask, i.e. around the bbox",
        )

        parser.add_argument(
            "--data_sanitize_paths",
            action="store_true",
            help="if true, wrong images or labels paths will be removed before training",
        )

        parser.add_argument(
            "--data_relative_paths",
            action="store_true",
            help="whether paths to images are relative to dataroot",
        )
        return parser

    def _gather_specific_options(self, opt, parser, args, flat_json):
        """
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        # modify model-related parser options
        model_name = opt.model_type
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        # parse again with new defaults
        opt = self._parse_args(parser, opt, args, flat_json)

        # modify dataset-related parser options
        dataset_name = opt.data_dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)
        opt = self._parse_args(parser, opt, args, flat_json)

        # save and return the parser
        return opt, parser

    def _after_parse(self, opt, set_device=True):
        super()._after_parse(opt, set_device)

        # set gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if set_device and len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_ids[0])

        # multimodal check
        if opt.model_multimodal:
            if not "cut" in opt.model_type:
                raise ValueError(
                    "Multimodal models are only supported with cut-based models at this stage, use --model_type accordingly"
                )
            if "resnet" in opt.G_netG:
                warnings.warn(
                    "ResNet encoder/decoder architectures do not mix well with multimodal training, use segformer or unet instead"
                )
            netE_size = int(opt.G_netE[-3:])
            if opt.data_crop_size != netE_size:
                msg = (
                    "latent multimodal decoder E has input size different than G output size: "
                    + str(netE_size)
                    + " vs "
                    + str(opt.data_crop_size)
                    + ", run may fail, use --G_netE accordingly"
                )
                warnings.warn(msg)

        # specific after parse
        opt = self._after_parse_specific(opt)

        # training is frequency space only available for a few architectures atm
        if opt.train_feat_wavelet:
            if not opt.G_netG in ["mobile_resnet_attn", "unet_mha", "uvit"]:
                raise ValueError(
                    "Wavelet training is only available for mobile_resnet_attn, unet_mha and uvit architectures"
                )

        # register options
        self.opt = opt

        return self.opt

    def _after_parse_specific(self, opt):
        """
        Add additional model-specific after parse function.
        These options are defined in the <after_parse> function
        in model classes.
        """
        # modify model-related parser options
        model_name = opt.model_type
        model_after_parse = models.get_after_parse(model_name)
        opt = model_after_parse(opt)

        return opt

    ### Functions related to CLI help
    # "alg" topic contains model options instead of filtered common options

    def topic_exists(self, topic):
        if topic is not None and topic.startswith("alg_"):
            return topic in self.get_topics("alg")
        else:
            return super().topic_exists(topic)

    def get_topics(self, topic=None):
        if topic == "alg":
            return {
                "alg_" + name: {"title": ""}
                for name in get_models_names()
                if name not in ["test", "template", "segmentation"]
            }
        else:
            return super().get_topics(topic)

    def get_topic_parser(self, topic):
        if topic is not None and topic.startswith("alg_"):
            model_name = topic[4:]
            parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS)
            return get_option_setter(model_name)(parser, True)
        else:
            return super().get_topic_parser(topic)
