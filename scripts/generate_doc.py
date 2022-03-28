#!/usr/bin/python3

import sys
import os
import argparse
from argparse import _HelpAction, _SubParsersAction, _StoreConstAction
import logging

jg_dir=os.path.join("/".join(os.path.abspath(__file__).split("/")[:-2]))
sys.path.append(jg_dir)

import options as opt

def main():
    parser = argparse.ArgumentParser(description="Generate documentation using the different model options")
    parser.add_argument('-v', "--verbose", action='store_true', help="Set logging level to INFO")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        
    options_md = """
# JoliGAN Options

Here are all the available options to call with `train.py`

"""
    options_md += document_parser(opt.get_parser())

    model_parsers = opt.get_models_parsers()
    model_names = list(model_parsers.keys())
    model_names.sort()
    
    for name in model_names:
        parser = model_parsers[name]
        options_md += "\n\n## %s\n\n%s" % (name, document_parser(parser))

    print(options_md)

    path_sv = os.path.join(jg_dir,'docs','options.md')
    with open(path_sv,'w+') as file:
        file.writelines(options_md)
    
# ====

# inspired by https://github.com/alex-rudakov/sphinx-argparse/blob/master/sphinxarg/parser.py (v0.2.5)
def document_parser(parser):
    help_str = "| Parameter | Type | Default | Description |\n"
    help_str += "| --- | --- | --- | --- |\n"

    for action in parser._get_positional_actions():
        if not isinstance(action, _SubParsersAction):
            continue
        for name, subaction in action._name_parser_map.items():
            pass
    
    for action_group in parser._action_groups:
        for action in action_group._group_actions:
            if isinstance(action, _HelpAction):
                continue

            name = ",".join(action.option_strings)

            if isinstance(action, _StoreConstAction):
                type_str = "flag"
                default = ""
            else:
                type_str = action.type.__name__ if action.type is not None else "str"
                default = action.default if action.default is not None else ""

            description = action.help if action.help is not None else ""
            description = description.replace("|", "\\|")

            help_str += "| %s | %s | %s | %s |\n" % (name, type_str, default, description)
    return help_str


if __name__ == "__main__":
    main()
