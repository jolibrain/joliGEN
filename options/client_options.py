from .train_options import TrainOptions
import argparse


class ClientOptions(TrainOptions):
    def initialize(self, parser):

        parser = TrainOptions.initialize(self, parser)

        parser.add_argument(
            "--method",
            type=str,
            default="launch_training",
            choices=["launch_training", "stop_training", "training_status"],
        )

        parser.add_argument(
            "--host",
            type=str,
            required=True,
            help="joligan server host",
        )

        parser.add_argument(
            "--port",
            type=int,
            required=True,
            help="joligan server post",
        )

        return parser
