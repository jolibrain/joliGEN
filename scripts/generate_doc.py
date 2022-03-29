#!/usr/bin/python3

import sys
import os
import argparse
from argparse import _HelpAction, _SubParsersAction, _StoreConstAction
import logging

jg_dir = os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]))
sys.path.append(jg_dir)

from options import TrainOptions


def main():
    parser = argparse.ArgumentParser(
        description="Generate documentation using the different model options"
    )
    parser.add_argument(
        "--save_to",
        default=os.path.join(jg_dir, "docs", "options.md"),
        help="Path of file to save",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Set logging level to INFO"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    options_md = """
# JoliGAN Options

Here are all the available options to call with `train.py`

"""
    schema = TrainOptions().get_schema()
    options_md += document_section(schema, 0, "--")

    print(options_md)

    if len(args.save_to) != 0:
        with open(args.save_to, "w+") as file:
            file.writelines(options_md)


def document_section(json_schema, level, opt_prefix):
    help_str = ""
    other_opt_str = "\n"

    for field_name in json_schema["properties"]:
        field = json_schema["properties"][field_name]

        if field["type"] == "object":
            other_opt_str += ((level + 2) * "#") + " " + field["description"] + "\n\n"
            other_opt_str += document_section(
                field, level + 1, opt_prefix + field_name + "_"
            )
        else:
            if len(help_str) == 0:
                help_str = "| Parameter | Type | Default | Description |\n"
                help_str += "| --- | --- | --- | --- |\n"

            type_str = field["type"]
            default_str = str(field["default"])

            if type_str == "boolean":
                type_str = "flag"
                default_str = ""
            elif type_str == "number":
                type_str = "float"
            elif type_str == "integer":
                type_str = "int"

            description = field["description"]
            if "enum" in field:
                description += "<br/><br/>_**Values:** "
                description += ", ".join([str(c) for c in field["enum"]])
                description += "_"

            help_str += "| %s | %s | %s | %s |\n" % (
                opt_prefix + field_name,
                type_str,
                default_str,
                description,
            )

    help_str += other_opt_str
    return help_str


if __name__ == "__main__":
    main()
