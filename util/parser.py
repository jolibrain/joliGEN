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

        if not "--dataroot" in remaining_args:
            remaining_args += ["--dataroot", "unused"]
        override_options_json = flatten_json(
            TrainOptions().parse_to_json(remaining_args)
        )

        with open(main_opt.config_json, "r") as jsonf:
            train_json = flatten_json(json.load(jsonf))

        for name in override_options_names:
            train_json[name] = override_options_json[name]

        opt = TrainOptions().parse_json(train_json, save_config=True)

        print("%s config file loaded" % main_opt.config_json)
    else:
        opt = TrainOptions().parse()  # get training options

    return opt
