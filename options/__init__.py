"""This package options includes option modules: training options, test options, and basic options (used in both training and test)."""
import argparse
from .base_options import BaseOptions
from .train_options import TrainOptions
from models import get_models_names, get_option_setter


def get_parser():
    """
    Create parser for documentation
    """
    parser = argparse.ArgumentParser()
    bo = BaseOptions()
    parser = bo.initialize(parser)
    return parser


def get_models_parsers():
    """
    Create a dict {model_name: parser} in which each parser hold arguments
    specific to a model
    """
    model_names = get_models_names()
    model_parsers = {}

    for model_name in model_names:
        parser = argparse.ArgumentParser()
        model_option_setter = get_option_setter(model_name)

        try:
            model_parsers[model_name] = model_option_setter(parser, True)
        except:
            pass

    return model_parsers
