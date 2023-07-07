from .train_options import TrainOptions
from util.util import MAX_INT


class PredictOptions(TrainOptions):
    """This class includes inference options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = TrainOptions.initialize(self, parser)

        parser.add_argument(
            "--model-in-file",
            type=str,
            help="file path to generator model (.pth file)",
            required=True,
        )

        parser.add_argument(
            "--img-in", type=str, help="image to transform", required=True
        )

        parser.add_argument(
            "--img-out", type=str, help="transformed image", required=True
        )

        parser.add_argument("--cpu", action="store_true", help="whether to use CPU")

        parser.add_argument(
            "--gpuid",
            type=int,
            default=0,
            help="which GPU to use",
        )

        # TODO: remove references in base_options.py
        # or replace by `isPredict`
        self.isTrain = True
        return parser
