from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument(
            "--test_ntest", type=int, default=float("inf"), help="# of test examples."
        )
        parser.add_argument(
            "--test_results_dir",
            type=str,
            default="./results/",
            help="saves results here.",
        )
        parser.add_argument(
            "--test_aspect_ratio",
            type=float,
            default=1.0,
            help="aspect ratio of result images",
        )
        # Dropout and Batchnorm has different behaviour during training and test.
        parser.add_argument(
            "--test_eval", action="store_true", help="use eval mode during test time."
        )
        parser.add_argument(
            "--test_num_test", type=int, default=50, help="how many test images to run"
        )
        # rewrite devalue values
        parser.set_defaults(model="test")
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default("crop_size"))
        self.isTrain = False
        return parser
