import argparse


class DiffusionOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument(
            "--model-in-file",
            help="file path to generator model (.pth file)",
            required=True,
        )

        self.parser.add_argument(
            "--img-width", default=-1, type=int, help="image width"
        )
        self.parser.add_argument(
            "--img-height", default=-1, type=int, help="image height"
        )

        self.parser.add_argument(
            "--crop-width",
            default=-1,
            type=int,
            help="crop width added on each side of the bbox (optional)",
        )
        self.parser.add_argument(
            "--crop-height",
            default=-1,
            type=int,
            help="crop height added on each side of the bbox (optional)",
        )

        self.parser.add_argument(
            "--bbox-width-factor",
            type=float,
            default=0.0,
            help="bbox width added factor of original width",
        )
        self.parser.add_argument(
            "--bbox-height-factor",
            type=float,
            default=0.0,
            help="bbox height added factor of original height",
        )

        self.parser.add_argument(
            "--sampling-steps", default=-1, type=int, help="number of sampling steps"
        )
        self.parser.add_argument(
            "--cpu", action="store_true", help="whether to use CPU"
        )
        self.parser.add_argument(
            "--gpuid", type=int, default=0, help="which GPU to use"
        )
        self.parser.add_argument(
            "--seed", type=int, default=-1, help="random seed for reproducibility"
        )

        self.parser.add_argument(
            "--dir-out", help="dir where img will be saved", required=True
        )

        self.parser.add_argument(
            "--mask_delta",
            type=int,
            default=[0],
            nargs="*",
            help="mask offset to allow generation of a bigger object, format : width (x) height (y) or only one size if square",
        )

        self.parser.add_argument(
            "--mask_square", action="store_true", help="whether to use square mask"
        )

        self.parser.add_argument("--name", help="generated img name", default="img")

        self.parser.add_argument(
            "--sampling_method",
            type=str,
            default="ddpm",
            choices=["ddpm", "ddim"],
            help="choose the sampling method between ddpm and ddim",
        )

        self.parser.add_argument(
            "--cls",
            type=int,
            default=-1,
            help="override input bbox classe for generation",
        )

    def parse(self):
        args = self.parser.parse_args()
        return args
