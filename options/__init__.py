"""This package options includes option modules: training options, test options, and basic options (used in both training and test)."""
import argparse
from .base_options import BaseOptions
from .train_options import TrainOptions
from .helpers import get_models_parsers


def get_parser():
    """
    Create parser for documentation
    """
    parser = argparse.ArgumentParser()
    bo = BaseOptions()
    parser = bo.initialize(parser)
    return parser
