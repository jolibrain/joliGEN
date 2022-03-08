import argparse
import os
import math
from collections import defaultdict
from util import util
import torch
import models
import data
from argparse import _HelpAction, _SubParsersAction, _StoreConstAction


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--ddp_port', type=str, default='12355')
        
        # model parameters
        parser.add_argument('--model_type', type=str, default='cycle_gan', help='chooses which model to use. [' + " | ".join(models.get_models_names()) + ']')
        parser.add_argument('--model_input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--model_output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--model_init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--model_init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        
        # generator
        parser.add_argument('--G_ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--G_netG', type=str, default='resnet_attn', help='specify generator architecture [resnet_9blocks | resnet_6blocks | resnet_3blocks | resnet_12blocks | mobile_resnet_9blocks | mobile_resnet_3blocks | resnet_attn | mobile_resnet_attn | unet_256 | unet_128 | stylegan2 | smallstylegan2 | segformer_attn_conv | segformer_conv]')
        parser.add_argument('--G_dropout', action='store_true', help='dropout for the generator')
        parser.add_argument('--G_spectral', action='store_true', help='whether to use spectral norm in the generator')
        parser.add_argument('--G_padding_type', type=str, help='whether to use padding in the generator, zeros or reflect', default='reflect')
        parser.add_argument('--G_norm', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
        parser.add_argument('--G_stylegan2_num_downsampling',
                            default=1, type=int,
                            help='Number of downsampling layers used by StyleGAN2Generator')
        parser.add_argument('--G_config_segformer',type=str,default='models/configs/segformer/segformer_config_b0.py',help='path to segforme configuration file')
        parser.add_argument('--G_attn_nb_mask_attn',default=10,type=int)
        parser.add_argument('--G_attn_nb_mask_input',default=1,type=int)
        
        # discriminator
        parser.add_argument('--D_ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--D_netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel] or any torchvision model [resnet18...]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--D_netD_global', type=str, default='none', help='specify discriminator architecture, any torchvision model can be used [resnet18...]. By default no global discriminator will be used.')
        parser.add_argument('--D_n_layers', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--D_norm', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for D')
        parser.add_argument('--D_dropout', action='store_true', help='whether to use dropout in the discriminator')
        parser.add_argument('--D_spectral', action='store_true', help='whether to use spectral norm in the discriminator')
        parser.add_argument('--D_projected_interp', type=int, default=-1, help='whether to force projected discriminator interpolation to a value > 224, -1 means no interpolation')
        parser.add_argument('--D_no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        parser.add_argument('--D_no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
        
        # semantic network
        parser.add_argument('--f_s_light',action='store_true', help='whether to use a light (unet) network for f_s')
        parser.add_argument('--f_s_dropout', action='store_true', help='dropout for the semantic network')
        parser.add_argument('--f_s_semantic_nclasses',default=2,type=int,help='number of classes of the semantic loss classifier')
        parser.add_argument('--f_s_semantic_threshold',default=1.0,type=float,help='threshold of the semantic classifier loss below with semantic loss is applied')
        parser.add_argument('--f_s_all_classes_as_one',action='store_true',help='if true, all classes will be considered as the same one (ie foreground vs background)')
        parser.add_argument('--f_s_nf', type=int, default=64, help='# of filters in the first conv layer of classifier')

        # dataset parameters
        parser.add_argument('--data_dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--data_direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--data_serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--data_num_threads', default=4, type=int, help='# threads for loading data')

        parser.add_argument('--data_load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--data_crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--data_max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--data_preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')




        # Online dataset creation options
        parser.add_argument('--data_online_creation_crop_size_A', type=int, default=512, help='crop to this size during online creation, it needs to be greater than bbox size for domain A')
        parser.add_argument('--data_online_creation_crop_delta_A', type=int, default=50, help='size of crops are random, values allowed are online_creation_crop_size more or less online_creation_crop_delta for domain A')
        parser.add_argument('--data_online_creation_mask_delta_A', type=int, default=0, help='mask offset to allow genaration of a bigger object in domain B (for semantic loss) for domain A')
        parser.add_argument('--data_online_creation_mask_square_A', action='store_true', help='whether masks should be squared for domain A')

        parser.add_argument('--data_online_creation_crop_size_B', type=int, default=512, help='crop to this size during online creation, it needs to be greater than bbox size for domain B')
        parser.add_argument('--data_online_creation_crop_delta_B', type=int, default=50, help='size of crops are random, values allowed are online_creation_crop_size more or less online_creation_crop_delta for domain B')
        parser.add_argument('--data_online_creation_mask_delta_B', type=int, default=0, help='mask offset to allow genaration of a bigger object in domain B (for semantic loss) for domain B')
        parser.add_argument('--data_online_creation_mask_square_B', action='store_true', help='whether masks should be squared for domain B')

        parser.add_argument('--data_sanitize_paths',action='store_true',help='if true, wrong images or labels paths will be removed before training')
        parser.add_argument('--data_relative_paths',action='store_true',help='whether paths to images are relative to dataroot')
        
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model_type
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.data_dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def to_json(self, ignore_default=False):
        """
        Converts an argparse namespace to a json-like dict containing the same arguments.
        This dict can be used to re-run with the same arguments from the API

        Parameters
            ignore_default Add only non-default options to json
        """
        json_args = dict()

        for k, v in sorted(vars(self.opt).items()):
            default = self.parser.get_default(k)
            if v != default or not ignore_default:
                json_args[k] = v

        return json_args

    def _after_parse(self, opt):
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        return self._after_parse(opt)

    def _json_parse_known_args(self, parser, opt, json_args):
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

                names = { action.dest }
                for opt_name in action.option_strings:
                    if opt_name.startswith("--"):
                        names.add(opt_name[2:])

                for name in names:
                    if name in json_args:
                        val = json_args[name]

                        if not isinstance(val, check_type):
                            raise ValueError("%s: Bad type" % (name,))

                        del json_args[name]

                setattr(opt, action.dest, val)

    def parse_json(self, json_args):
        """
        Parse a json-like dict using the joliGAN argument parser.

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

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt = argparse.Namespace()
        self._json_parse_known_args(parser, opt, json_args)

        model_name = opt.model_type
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        self._json_parse_known_args(parser, opt, json_args)
        self.parser = parser

        if len(json_args) != 0:
            raise ValueError("%d remaining keys in json args: %s" % (len(json_args), ",".join(json_args.keys())))

        return self._after_parse(opt)

    def get_schema(self, allow_nan = False):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt = argparse.Namespace()
        self._json_parse_known_args(parser, opt, {})
        
        named_parsers = {"": parser}
        for model_name in models.get_models_names():
            if self.isTrain and model_name in ["test"]:
                continue
            setter = models.get_option_setter(model_name)
            model_parser = argparse.ArgumentParser()
            setter(model_parser)
            self._json_parse_known_args(model_parser, opt, {})
            named_parsers[model_name] = model_parser

        self.opt = opt
        self.parser = parser
        json_vals = self.to_json()

        for k in json_vals:
            if json_vals[k] is None:
                json_vals[k] = "None"
            if not allow_nan:
                if json_vals[k] == float("inf"):
                    json_vals[k] = 1e100
                if json_vals[k] == float("-inf"):
                    json_vals[k] = -1e100
                if type(json_vals[k]) == float and math.isnan(json_vals[k]):
                    json_vals[k] = 0

        from pydantic import create_model
        schema = create_model(type(self).__name__, **json_vals).schema()
        
        option_tags = defaultdict(list)

        for parser_name in named_parsers:
            current_parser = named_parsers[parser_name]

            for action_group in current_parser._action_groups:
                for action in action_group._group_actions:
                    if isinstance(action, _HelpAction):
                        continue

                    if len(parser_name) > 0:
                        option_tags[action.dest].append(parser_name)

                    if action.dest in schema["properties"]:
                        field = schema["properties"][action.dest]
                        description = action.help if action.help is not None else ""
                        for c in "#*<>":
                            description = description.replace(c, "\\" + c)
                        field["description"] = description
                        if "title" in field:
                            del field["title"]

        for tagged in option_tags:
            tags = " | ".join(option_tags[tagged])
            schema["properties"][tagged]["description"] = "[" + tags + "]\n\n" + schema["properties"][tagged]["description"]

        return schema
