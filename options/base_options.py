import argparse
import os
import math
from collections import defaultdict
from util import util
import torch
import models
import data
from argparse import _HelpAction, _SubParsersAction, _StoreConstAction
from util.util import MAX_INT, flatten_json
import json
from models.modules.classifiers import TORCH_MODEL_CLASSES
import warnings

TRAIN_JSON_FILENAME = "train_config.json"


class BaseOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    opt_schema = {
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
                "properties": {"online_creation": {"title": "Online created datasets"}},
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
    general_options = ["model_type"]

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
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
            choices=["cut", "cycle_gan", "palette"],
            help="chooses which model to use.",
        )
        parser.add_argument(
            "--model_input_nc",
            type=int,
            default=3,
            choices=[1, 3],
            help="# of input image channels: 3 for RGB and 1 for grayscale",
        )
        parser.add_argument(
            "--model_output_nc",
            type=int,
            default=3,
            choices=[1, 3],
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
            help="path to sam weight for D, e.g. models/configs/sam/pretrain/sam_vit_b_01ec64.pth",
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
                "resnet_9blocks",
                "resnet_6blocks",
                "resnet_3blocks",
                "resnet_12blocks",
                "mobile_resnet_9blocks",
                "mobile_resnet_3blocks",
                "resnet_attn",
                "mobile_resnet_attn",
                "unet_256",
                "unet_128",
                "stylegan2",
                "smallstylegan2",
                "segformer_attn_conv",
                "segformer_conv",
                "ittr",
                "unet_mha",
                "uvit",
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
            "--G_stylegan2_num_downsampling",
            default=1,
            type=int,
            help="Number of downsampling layers used by StyleGAN2Generator",
        )
        parser.add_argument(
            "--G_config_segformer",
            type=str,
            default="models/configs/segformer/segformer_config_b0.json",
            help="path to segformer configuration file for G",
        )
        parser.add_argument("--G_attn_nb_mask_attn", default=10, type=int)

        parser.add_argument("--G_attn_nb_mask_input", default=1, type=int)

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

        parser.add_argument("--G_unet_mha_num_head_channels", default=32, type=int)
        parser.add_argument("--G_unet_mha_num_heads", default=1, type=int)

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
            "-G_unet_mha_attn_res",
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
                "stylegan2",
                "patchstylegan2",
                "smallpatchstylegan2",
                "projected_d",
                "temporal",
                "vision_aided",
                "depth",
                "mask",
                "sam",
            ]
            + list(TORCH_MODEL_CLASSES.keys()),
            help="specify discriminator architecture, D_n_layers allows you to specify the layers in the discriminator. NB: duplicated arguments will be ignored.",
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
            "--D_temporal_number_frames",
            type=int,
            default=5,
            help="how many successive frames use for temporal loss",
        )

        parser.add_argument(
            "--D_temporal_frame_step",
            type=int,
            default=30,
            help="how many frames between successive frames selected",
        )

        parser.add_argument("--D_temporal_every", type=int, default=4)
        parser.add_argument(
            "--D_temporal_num_common_char",
            type=int,
            default=-1,
            help="how many characters (the first ones) are used to identify a video; if =-1 natural sorting is used ",
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
            help="path to sam weight for f_s, e.g. models/configs/sam/pretrain/sam_vit_b_01ec64.pth",
        )

        # cls semantic network
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
            default="models/configs/segformer/segformer_config_b0.py",
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
                "temporal",
                "self_supervised_temporal",
                "single",
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
            type=int,
            default=[0],
            nargs="*",
            help="ratio mask offset to allow generation of a bigger object in domain B (for semantic loss) for domain A, format : width (x) height (y) or only one size if square",
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
            type=int,
            default=[0],
            nargs="*",
            help="mask offset to allow genaration of a bigger object in domain B (for semantic loss) for domain B, format : width (y) height (x) or only one size if square",
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

        self.initialized = True
        return parser

    def gather_options(self, args=None):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args(args)

        # modify model-related parser options
        model_name = opt.model_type
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args(args)  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.data_dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args(args)

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, "{}_opt.txt".format(opt.phase))
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def _split_key(self, key, schema):
        """
        Converts argparse option key to json path
        (ex: data_online_creation_crop_delta_A will be converted to
        ("data", "online_creation", "delta_A")
        """
        # General options are always in the root of the json
        if key in self.general_options:
            return (key,)

        if "properties" in schema:
            for prop in schema["properties"]:
                if key.startswith(prop + "_"):
                    nested_keys = self._split_key(
                        key[len(prop) + 1 :], schema["properties"][prop]
                    )
                    return (prop, *nested_keys)
        return (key,)

    def to_json(self, ignore_default=False):
        """
        Converts an argparse namespace to a json-like dict containing the same arguments.
        This dict can be used to re-run with the same arguments from the API

        Parameters
            ignore_default Add only non-default options to json
        """

        def schema_to_json(schema):
            json_args = {}
            if "properties" in schema:
                for key in schema["properties"]:
                    json_args[key] = schema_to_json(schema["properties"][key])
            return json_args

        json_args = schema_to_json(self.opt_schema)

        for k, v in sorted(vars(self.opt).items()):
            default = self.parser.get_default(k)

            if v != default or not ignore_default:
                path = self._split_key(k, self.opt_schema)
                parent = json_args
                for cat in path[:-1]:
                    parent = parent[cat]
                parent[path[-1]] = v

        return dict(json_args)

    def _after_parse(self, opt, set_device=True):
        if hasattr(self, "isTrain"):
            opt.isTrain = self.isTrain  # train or test
        else:
            opt.isTrain = False
            self.isTrain = False

        # process opt.suffix
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

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

        # bbox selection check
        if opt.data_online_select_category != -1 and not opt.data_sanitize_paths:
            raise ValueError(
                "Bounding box class selection requires --data_sanitize_paths"
            )

        # vitclip16 projector only works with input size 224
        if opt.D_proj_network_type == "vitclip16":
            if opt.D_proj_interp != 224:
                warnings.warn(
                    "ViT-B/16 (vitclip16) projector only works with input size 224, setting D_proj_interp to 224"
                )
            opt.D_proj_interp = 224

        # Dsam requires D_weight_sam
        if "sam" in opt.D_netDs and opt.D_weight_sam == "":
            raise ValueError(
                "Dsam requires D_weight_sam, please specify a path to a pretrained sam model"
            )

        # diffusion D + vitsmall check
        if opt.dataaug_D_diffusion and "vit" in opt.D_proj_network_type:
            raise ValueError(
                "ViT type projectors are not yet compatible with diffusion augmentation at discriminator level"
            )

        # sam with bbox prompting requires Pytorch 2
        if torch.__version__[0] != "2":
            if (
                opt.f_s_net == "sam"
                and opt.data_dataset_mode == "unaligned_labeled_mask_online"
            ):
                raise ValueError("SAM with masks and bbox prompting requires Pytorch 2")
        if opt.f_s_net == "sam" and opt.data_dataset_mode == "unaligned_labeled_mask":
            raise warning.warn("SAM with direct masks does not use mask/bbox prompting")

        self.opt = opt

        return self.opt

    def save_options(self):
        self.print_options(self.opt)
        with open(
            os.path.join(self.opt.checkpoints_dir, self.opt.name, TRAIN_JSON_FILENAME),
            "w+",
        ) as outfile:
            json.dump(self.to_json(), outfile, indent=4)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        self.opt = self.gather_options()
        self.save_options()
        opt = self._after_parse(self.opt)
        return opt

    def parse_to_json(self, args=None):
        self.opt = self.gather_options(args)
        return self.to_json()

    def _json_parse_known_args(self, parser, opt, json_args):
        """
        json_args: flattened json of train options
        """
        for action_group in parser._action_groups:
            for action in action_group._group_actions:
                if isinstance(action, _HelpAction):
                    continue

                if hasattr(opt, action.dest):
                    # already parsed
                    continue

                if isinstance(action, _StoreConstAction):
                    val = False
                    check_type = bool
                else:
                    val = action.default
                    check_type = action.type

                if check_type is None:
                    check_type = str
                elif check_type is util.str2bool:
                    check_type = bool

                names = {action.dest}
                for opt_name in action.option_strings:
                    if opt_name.startswith("--"):
                        names.add(opt_name[2:])

                for name in names:
                    if name in json_args:
                        val = json_args[name]

                        if (
                            type(val) == int and check_type == float
                        ):  # int are considered as float
                            val = float(val)

                        elif action.nargs == "+" or action.nargs == "*":
                            if not isinstance(val, list) or not all(
                                isinstance(elt, check_type) for elt in val
                            ):
                                raise ValueError(
                                    "%s: Bad type (%s, should be list of %s)"
                                    % (name, str(type(val)), str(check_type))
                                )

                        elif not isinstance(val, check_type):
                            raise ValueError(
                                "%s: Bad type (%s, should be %s)"
                                % (name, str(type(val)), str(check_type))
                            )

                        del json_args[name]

                setattr(opt, action.dest, val)

    def parse_json(self, json_args, save_config=False, set_device=True):
        """
        Parse a json-like dict using the joliGEN argument parser.

        JSON structure is

        ```
        {
            "base_option1": ...,
            "base_option2": ...,
            "cut_option1": ...,
            ...
        }
        ```
        """

        if not hasattr(self, "isTrain"):
            self.isTrain = False

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        flat_json = flatten_json(json_args)

        opt = argparse.Namespace()
        self._json_parse_known_args(parser, opt, flat_json)

        model_name = opt.model_type
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        self._json_parse_known_args(parser, opt, flat_json)
        self.parser = parser

        if len(flat_json) != 0:
            # raise ValueError("%d remaining keys in json args: %s" % (len(json_args), ",".join(json_args.keys())))
            print(
                "%d remaining keys in json args: %s"
                % (len(flat_json), ",".join(flat_json.keys()))
            )  # it's not an error anymore because server launching is done with all of the options even those from other models, raising an error will lead to a server crash

        if save_config:
            self.opt = opt
            self.save_options()

        return self._after_parse(opt, set_device)

    def get_schema(self, allow_nan=False):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        opt = argparse.Namespace()
        self._json_parse_known_args(parser, opt, {})

        named_parsers = {"": parser}
        for model_name in models.get_models_names():
            if self.isTrain and model_name in ["test", "template"]:
                continue
            setter = models.get_option_setter(model_name)
            model_parser = argparse.ArgumentParser()
            setter(model_parser)
            self._json_parse_known_args(model_parser, opt, {})
            named_parsers[model_name] = model_parser

        self.opt = opt
        self.parser = parser
        json_vals = self.to_json()

        from pydantic import create_model

        def json_to_schema(name, json_vals, schema_tmplate):
            for k in json_vals:
                if json_vals[k] is None:
                    json_vals[k] = "None"
                if not allow_nan:
                    if type(json_vals[k]) == float and math.isnan(json_vals[k]):
                        json_vals[k] = 0

            schema = create_model(name, **json_vals).schema()

            if "description" in schema_tmplate:
                schema["description"] = schema_tmplate["description"]
            if "title" in schema_tmplate:
                schema["title"] = schema_tmplate["title"]
            if "properties" in schema_tmplate:
                for prop in schema_tmplate["properties"]:
                    schema["properties"][prop] = json_to_schema(
                        prop, json_vals[prop], schema_tmplate["properties"][prop]
                    )

            return schema

        schema = json_to_schema(type(self).__name__, json_vals, self.opt_schema)

        for parser_name in named_parsers:
            current_parser = named_parsers[parser_name]

            for action_group in current_parser._action_groups:
                for action in action_group._group_actions:
                    if isinstance(action, _HelpAction):
                        continue

                    path = self._split_key(action.dest, self.opt_schema)
                    field = schema
                    for cat in path:
                        if "properties" not in field or cat not in field["properties"]:
                            field = None
                            break
                        field = field["properties"][cat]

                    if field is not None:
                        description = action.help if action.help is not None else ""
                        for c in "#*<>":
                            description = description.replace(c, "\\" + c)
                        field["description"] = description

                        if action.nargs == "+":
                            field["items"]["enum"] = action.choices
                            if isinstance(action.default[0], str):
                                cur_type = "string"
                            field["items"]["type"] = cur_type

                        elif action.choices:
                            field["enum"] = action.choices

                        if "title" in field:
                            del field["title"]

        return schema
