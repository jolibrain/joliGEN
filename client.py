# -*- coding: utf-8 -*-
#
# JoliGAN Python client
#
# Licence:
#
# Copyright 2020-2022 Jolibrain SASU

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Here are calls examples that you can use to make API calls to a JoliGAN server. Please note that you have to run a server first.

#Launch a training
python client.py --host jg_server_host --port jg_server_port [joligan commandline options eg --dataroot /path/to/data --model_type cut]
NB: the name given in joligan commandline options will also be the name of the training process.

# List trainings in progress
python client.py --method training_status --host jg_server_host --port jg_server_port

# Stop a training
python client.py --method training_status --host jg_server_host --port jg_server_port --name training_name
"""

import requests
from options.train_options import TrainOptions
import sys
import argparse
import json


def train(host: str, port: int, name: str, train_options: dict):
    json_opt = {}
    json_opt["sync"] = False
    train_options["name"] = name
    json_opt["train_options"] = train_options

    url = "http://%s:%d" % (host, port) + "/train/%s" % name

    x = requests.post(url=url, json=json_opt)

    print("Training %s started." % x.json()["name"])


def delete(host: str, port: int, name: str):
    url = "http://%s:%d" % (host, port) + "/train/%s" % name
    x = requests.delete(url=url)

    print("Training %s has been stopped." % x.json()["name"])


def get_status(host: str, port: int):
    url = "http://%s:%d" % (host, port) + "/train"

    x = requests.get(url=url)

    print("There are %i trainings in progress." % (len(x.json()["processes"])))

    for process in x.json()["processes"]:
        print("Name: %s, status: %s" % (process["name"], process["status"]))


def main_client(args: list):
    main_parser = argparse.ArgumentParser()

    main_parser.add_argument(
        "--config_json", type=str, default="", help="path to json config"
    )
    main_parser.add_argument(
        "--method",
        type=str,
        default="launch_training",
        choices=["launch_training", "stop_training", "training_status"],
    )

    main_parser.add_argument(
        "--name",
        type=str,
        default="training_name",
    )

    main_parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="joligan server host",
    )

    main_parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="joligan server post",
    )

    main_opt, args = main_parser.parse_known_args(args)

    host = main_opt.host
    port = main_opt.port
    method = main_opt.method
    name = main_opt.name

    if method == "launch_training":
        if main_opt.config_json != "":
            with open(main_opt.config_json, "r") as jsonf:
                train_options = json.load(jsonf)
            print("%s config file loaded" % main_opt.config_json)
        else:
            train_options = TrainOptions().parse_to_json(args)

        train(host, port, name, train_options)

    elif method == "stop_training":
        delete(host, port, name)

    elif method == "training_status":
        get_status(host, port)
    else:
        raise


if __name__ == "__main__":
    args = sys.argv[1:]  # removing the script name
    main_client(args)
