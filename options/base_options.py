import argparse
import json
import math
import os
import warnings
from copy import deepcopy
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

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False
        self.opt_schema = {}
        self.general_options = []

    def initialize(self, parser):
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

        self.initialized = True
        return parser

    def gather_options(self, args=None, json_args=None):
        """Initialize our parser with options (only once).

        Parameters:
            args command line arguments, if None, using sys.argv
            json_args json containing arguments. If not None, using json_args in
            place of args.
        """
        if not hasattr(self, "isTrain"):
            self.isTrain = False

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                add_help=False,
            )
            # set_custom_help(parser)
            parser = self.initialize(parser)

        flat_json = None
        if json_args is not None:
            flat_json = flatten_json(json_args)

        # get the basic options
        opt = self._parse_args(parser, None, args, flat_json)

        # get specific options
        opt, parser = self._gather_specific_options(opt, parser, args, flat_json)

        # save and return the parser
        self.parser = parser
        return opt

    def _gather_specific_options(self, opt, parser, args, flat_json):
        """
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        return opt, parser

    def _parse_args(self, parser, opt, args, flat_json, only_known=True):
        """
        Parameters:
            args Command line arguments as an array of strings, or None
            flat_json If not None, arguments will be parsed from flattened json
        """
        if flat_json is not None:
            if opt is None:
                opt = argparse.Namespace()
            self._json_parse_known_args(parser, opt, flat_json)

            if not only_known:
                if len(flat_json) != 0:
                    # raise ValueError("%d remaining keys in json args: %s" % (len(json_args), ",".join(json_args.keys())))
                    print(
                        "%d remaining keys in json args: %s"
                        % (len(flat_json), ",".join(flat_json.keys()))
                    )  # it's not an error anymore because server launching is done with all of the options even those from other models, raising an error will lead to a server crash
        else:
            if only_known:
                opt, _ = parser.parse_known_args(args)
            else:
                opt = parser.parse_args(args)
        return opt

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

        def order_like_schema(json_args, schema):
            """
            Put keys in the same order as templated schema.
            It makes browsing the json easier and has an impact on documentation organization
            """
            if "properties" in schema:
                for key in schema["properties"]:
                    if key in json_args:
                        new_entry = order_like_schema(
                            json_args[key], schema["properties"][key]
                        )
                        # readd = reorder
                        del json_args[key]
                        json_args[key] = new_entry
            return json_args

        json_args = {}

        for k, v in sorted(vars(self.opt).items()):
            default = self.parser.get_default(k)

            if v != default or not ignore_default:
                path = self._split_key(k, self.opt_schema)
                parent = json_args
                for cat in path[:-1]:
                    if cat not in parent:
                        parent[cat] = {}
                    parent = parent[cat]

                parent[path[-1]] = v

        json_args = order_like_schema(json_args, self.opt_schema)
        return dict(json_args)

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

        # register options
        self.opt = opt

        return self.opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        self.opt = self.gather_options()
        self.save_options()
        opt = self._after_parse(self.opt)
        return opt

    def parse_to_json(self, args=None):
        self.opt = self.gather_options(args)
        return self.to_json()

    def parse_json(self, json_args, save_config=False, set_device=True):
        """
        Parse a json-like dict using the joliGEN argument parser.

        JSON structure can be flattened like this:

        ```
        {
            "base_option1": ...,
            "base_option2": ...,
            "cut_option1": ...,
            ...
        }
        ```
        or it can use categories:
        ```
        {
            "base": {
                "option1": ...,
                "option2": ...,
            },
            {
                "cut": ...
            },
            ...
        }
        ```
        """
        opt = self.gather_options(json_args=json_args)

        if save_config:
            self.opt = opt
            self.save_options()
        return self._after_parse(opt, set_device)

    # TODO only works with common options, not with inference options
    def save_options(self):
        self.print_options(self.opt)
        with open(
            os.path.join(self.opt.checkpoints_dir, self.opt.name, TRAIN_JSON_FILENAME),
            "w+",
        ) as outfile:
            json.dump(self.to_json(), outfile, indent=4)

    def get_schema(self, allow_nan=False, model_names=None):
        """
        Generate a pydantic schema of the options. This schema will be used
        in server documentation and input validation from the server.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        opt = argparse.Namespace()
        self._json_parse_known_args(parser, opt, {})

        named_parsers = {"": parser}
        if model_names is None:
            model_names = models.get_models_names()

        for model_name in model_names:
            if self.isTrain and model_name in ["test", "template"]:
                continue
            model_setter = models.get_option_setter(model_name)
            model_parser = argparse.ArgumentParser()
            model_setter(model_parser)
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
                            cur_type = None
                            if isinstance(action.default[0], str):
                                cur_type = "string"
                            elif isinstance(action.default[0], list):
                                cur_type = "list"
                            if cur_type is not None:
                                field["items"]["type"] = cur_type

                        elif action.choices:
                            field["enum"] = action.choices

                        if "title" in field:
                            del field["title"]

        return schema
