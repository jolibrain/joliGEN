from .train_options import TrainOptions


class EvaluationOptions(TrainOptions):
    """This class includes evaluation options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = TrainOptions.initialize(self, parser)

        parser.add_argument(
            "--eval_dataset_sizes",
            type=str,
            default="100,250,500,750,1000",
            help="different dataset sizes for evaluation",
        )

        self.isTrain = True
        return parser
