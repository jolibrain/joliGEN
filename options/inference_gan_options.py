from options import BaseOptions


class InferenceGANOptions(BaseOptions):
    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser = super().initialize(parser)

        parser.add_argument(
            "--model_in_file",
            help="file path to generator model (.pth file)",
            required=True,
        )

        parser.add_argument("--img_in", help="image to transform", required=True)
        parser.add_argument("--img_out", help="transformed image", required=True)
        parser.add_argument(
            "--img_width", type=int, help="image width, defaults to model crop size"
        )
        parser.add_argument(
            "--img_height", type=int, help="image height, defaults to model crop size"
        )
        parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
        parser.add_argument("--gpuid", type=int, default=0, help="which GPU to use")
        parser.add_argument(
            "--compare",
            action="store_true",
            help="Concatenate the true image and the transformed image",
        )

        self.isTrain = False
        return parser
