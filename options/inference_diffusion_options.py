from util.util import pairs_of_floats, pairs_of_ints

from .base_options import BaseOptions
from util.util import MAX_INT


class InferenceDiffusionOptions(BaseOptions):
    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser = super().initialize(parser)

        # parser = self.initialize_mutable(parser)

        self.initialized = True

        return parser

    def initialize_mutable(self, parser):

        parser = super().initialize_mutable(parser)

        parser.add_argument(
            "--model_in_file",
            help="file path to generator model (.pth file)",
            required=True,
        )

        parser.add_argument("--img_width", default=-1, type=int, help="image width")
        parser.add_argument("--img_height", default=-1, type=int, help="image height")

        parser.add_argument(
            "--crop_width",
            default=-1,
            type=int,
            help="crop width added on each side of the bbox (optional)",
        )
        parser.add_argument(
            "--crop_height",
            default=-1,
            type=int,
            help="crop height added on each side of the bbox (optional)",
        )

        parser.add_argument(
            "--bbox_width_factor",
            type=float,
            default=0.0,
            help="bbox width added factor of original width",
        )
        parser.add_argument(
            "--bbox_height_factor",
            type=float,
            default=0.0,
            help="bbox height added factor of original height",
        )

        parser.add_argument(
            "--sampling_steps", default=-1, type=int, help="number of sampling steps"
        )
        parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
        parser.add_argument(
            "--gpuid", type=int, default=0, help="which GPU to use, if -1 cpu is used"
        )
        parser.add_argument(
            "--seed", type=int, default=-1, help="random seed for reproducibility"
        )

        parser.add_argument(
            "--dir_out", help="dir where img will be saved", required=True
        )

        parser.add_argument(
            "--mask_delta",
            default=[[0]],
            nargs="+",
            type=pairs_of_ints,
            help="mask offset to allow generation of a bigger object, format : width (x) height (y) for each class or only one size if square",
        )
        parser.add_argument(
            "--mask_delta_ratio",
            default=[[0]],
            nargs="+",
            type=pairs_of_floats,
            help="ratio mask offset to allow generation of a bigger object, format : width (x),height (y) for each class or only one size if square",
        )

        parser.add_argument(
            "--mask_square", action="store_true", help="whether to use square mask"
        )

        parser.add_argument("--name", help="generated img name", default="img")

        parser.add_argument(
            "--sampling_method",
            type=str,
            default="ddpm",
            choices=["ddpm", "ddim"],
            help="choose the sampling method between ddpm and ddim",
        )

        parser.add_argument(
            "--cls_value",
            type=int,
            default=-1,
            help="override input bbox classe for generation",
        )

        # last options

        parser.add_argument("--img_in", help="image to transform", required=True)
        parser.add_argument("--previous-frame", help="image to transform", default=None)
        parser.add_argument(
            "--mask-in", help="mask used for image transformation", required=False
        )
        parser.add_argument("--bbox_in", help="bbox file used for masking")

        parser.add_argument(
            "--nb_samples", help="nb of samples generated", type=int, default=1
        )
        parser.add_argument(
            "--bbox_ref_id", help="bbox id to use", type=int, default=-1
        )
        parser.add_argument("--cond_in", help="conditionning image to use")
        parser.add_argument("--cond_keep_ratio", action="store_true")
        parser.add_argument("--cond_rotation", type=float, default=0)
        parser.add_argument("--cond_persp_horizontal", type=float, default=0)
        parser.add_argument("--cond_persp_vertical", type=float, default=0)

        parser.add_argument(
            "--alg_palette_guidance_scale",
            type=float,
            default=0.0,  # literature value: 0.2
            help="scale for classifier-free guidance, default is conditional DDPM only",
        )
        parser.add_argument(
            "--alg_palette_sketch_canny_thresholds",
            type=int,
            nargs="+",
            default=[0, 255 * 3],
            help="Canny thresholds",
        )
        parser.add_argument(
            "--alg_palette_super_resolution_downsample",
            action="store_true",
            help="whether to downsample the image for super resolution",
        )
        parser.add_argument(
            "--min_crop_bbox_ratio",
            type=float,
            help="minimum crop/bbox ratio, allows to add context when bbox is larger than crop",
        )

        self.isTrain = False

        return parser
