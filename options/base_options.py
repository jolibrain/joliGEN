import argparse
import json
import math
import os
import warnings
from argparse import _HelpAction, _StoreConstAction, _SubParsersAction
from collections import defaultdict

import torch

import data
import models
from util import util
from util.util import MAX_INT, flatten_json, pairs_of_floats, pairs_of_ints

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
        print("initialize")
        parser = self.initialize_mutable(parser)
        parser = self.initialize_static(parser)
        return parser

    def initialize_mutable(self, parser):
        # basic parameters
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

        # dataset parameters
        parser.add_argument(
            "--data_load_size", type=int, default=286, help="scale images to this size"
        )
        parser.add_argument(
            "--data_crop_size", type=int, default=256, help="then crop to this size"
        )
        parser.add_argument(
            "--data_refined_mask",
            action="store_true",
            help="whether to use refined mask with sam",
        )

        # data temporal options
        parser.add_argument(
            "--data_temporal_num_common_char",
            type=int,
            default=-1,
            help="how many characters (the first ones) are used to identify a video; if =-1 natural sorting is used ",
        )

        return parser

    def initialize_static(self, parser):
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
            "--model_prior_321_backwardcompatibility",
            action="store_true",
            help="whether to load models from previous version of JG.",
        )
        parser.add_argument(
            "--model_multimodal",
            action="store_true",
            help="multimodal model with random latent input vector",
        )
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

        # unet or uvit specific options
        parser.add_argument("--G_unet_mha_num_head_channels", default=32, type=int)
        parser.add_argument("--G_unet_mha_num_heads", default=1, type=int)
        parser.add_argument(
            "--G_uvit_num_transformer_blocks",
            default=6,
            type=int,
            help="Number of transformer blocks in UViT",
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

        # G diff params
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

        # mask semantic network : f_s
        parser.add_argument(
            "--f_s_semantic_nclasses",
            default=2,
            type=int,
            help="number of classes of the semantic loss classifier",
        )

        parser.add_argument(
            "--cls_semantic_nclasses",
            default=2,
            type=int,
            help="number of classes of the semantic loss classifier",
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

        # other data options
        parser.add_argument(
            "--data_inverted_mask",
            action="store_true",
            help="whether to invert the mask, i.e. around the bbox",
        )

        return parser

    def gather_specific_options(self, opt, parser, args):
        """
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        # modify model-related parser options
        model_name = opt.model_type
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args(args)  # parse again with new defaults

        return parser

    def gather_options(self, args=None):
        """Initialize our parser with basic options(only once)."""
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args(args)

        # get specific options
        parser = self.gather_specific_options(opt=opt, parser=parser, args=args)

        # save and return the parser
        self.parser = parser
        return parser

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

    def _after_parse(self, opt, set_device=True):
        if hasattr(self, "isTrain"):
            opt.isTrain = self.isTrain  # train or test
        else:
            opt.isTrain = False
            self.isTrain = False

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
        self.opt = self.gather_options().parse_args(args=None)
        self.save_options()
        opt = self._after_parse(self.opt)
        return opt

    def parse_to_json(self, args=None):
        self.opt = self.gather_options(args).parse_args(args=args)
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

                special_type = False

                if check_type is None:
                    check_type = str
                elif check_type is util.str2bool:
                    check_type = bool
                elif (
                    check_type is util.pairs_of_floats
                    or check_type is util.pairs_of_ints
                ):
                    check_type = list
                    special_type = True

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
                            if (
                                not isinstance(val, list)
                                or (not all(isinstance(elt, check_type) for elt in val))
                            ) and not special_type:
                                print(val)
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
        print("model_name", model_name)
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

        def replace_item(obj, existing_value, replace_value, allow_nan):
            for k, v in obj.items():
                if isinstance(v, dict):
                    obj[k] = replace_item(v, existing_value, replace_value, allow_nan)
                if v == existing_value:
                    obj[k] = replace_value
                if not allow_nan:
                    if type(obj[k]) == float and math.isnan(obj[k]):
                        obj[k] = 0
            return obj

        def json_to_schema(name, json_vals, schema_tmplate):
            replace_item(json_vals, None, "None", allow_nan=allow_nan)

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
                            elif isinstance(action.default[0], list):
                                cur_type = "list"
                            print("field", field)
                            field["items"]["type"] = cur_type

                        elif action.choices:
                            field["enum"] = action.choices

                        if "title" in field:
                            del field["title"]

        return schema
