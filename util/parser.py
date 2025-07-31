import json
import os
import torch
from util.util import flatten_json
from options.train_options import TrainOptions


def get_override_options_names(remaining_args):
    return_options_names = []

    for arg in remaining_args:
        if arg.startswith("--"):
            return_options_names.append(arg[2:])

    return return_options_names


def get_opt(main_opt, remaining_args):
    if main_opt.config_json != "":
        override_options_names = get_override_options_names(remaining_args)

        with open(main_opt.config_json, "r") as jsonf:
            train_json = flatten_json(json.load(jsonf))

        # Save the config file when --save_config is passed
        is_config_saved = False
        if "save_config" in override_options_names:
            is_config_saved = True
            override_options_names.remove("save_config")
            remaining_args.remove("--save_config")

        if not "--dataroot" in remaining_args:
            remaining_args += ["--dataroot", "unused"]
        # model_type is mandatory to load the correct options
        remaining_args += ["--model_type", train_json["model_type"]]
        override_options_json = flatten_json(
            TrainOptions().parse_to_json(remaining_args)
        )

        for name in override_options_names:
            train_json[name] = override_options_json[name]

        opt = TrainOptions().parse_json(train_json, save_config=is_config_saved)

        print("%s config file loaded" % main_opt.config_json)
    else:
        opt = TrainOptions().parse()  # get training options

    return opt
