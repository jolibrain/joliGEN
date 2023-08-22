from .base_options import BaseOptions
from util.util import MAX_INT, pairs_of_floats, pairs_of_ints


class PredictOptions(BaseOptions):
    """This class includes inference options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument(
            "--model-in-file",
            type=str,
            help="file path to generator model (.pth file)",
            required=True,
        )

        parser.add_argument(
            "--predict-method",
            type=str,
            help="Python script filename to load from $JOLIGEN/scripts/ folder",
        )

        parser.add_argument(
            "--seed", type=int, help="Torch seed, desactivated by default", default=0
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

        # Diffusion arguments

        parser.add_argument("--previous-frame", help="image to transform", default=None)
        parser.add_argument(
            "--mask-in", help="mask used for image transformation", required=False
        )
        parser.add_argument("--bbox-in", help="bbox file used for masking")

        parser.add_argument(
            "--nb_samples", help="nb of samples generated", type=int, default=1
        )
        parser.add_argument(
            "--bbox_ref_id", help="bbox id to use", type=int, default=-1
        )
        parser.add_argument("--cond-in", help="conditionning image to use")
        parser.add_argument("--cond_keep_ratio", action="store_true")
        parser.add_argument("--cond_rotation", type=float, default=0)
        parser.add_argument("--cond_persp_horizontal", type=float, default=0)
        parser.add_argument("--cond_persp_vertical", type=float, default=0)
        parser.add_argument(
            "--alg_palette_cond_image_creation",
            type=str,
            choices=[
                "y_t",
                "previous_frame",
                "sketch",
                "canny",
                "depth",
                "hed",
                "hough",
                "low_res",
                "sam",
            ],
            help="how cond_image is created",
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
            "--alg_palette_sam_use_gaussian_filter",
            action="store_true",
            default=False,
            help="whether to apply a gaussian blur to each SAM masks",
        )

        parser.add_argument(
            "--alg_palette_sam_no_sobel_filter",
            action="store_false",
            default=True,
            help="whether to not use a Sobel filter on each SAM masks",
        )

        parser.add_argument(
            "--alg_palette_sam_no_output_binary_sam",
            action="store_false",
            default=True,
            help="whether to not output binary sketch before Canny",
        )

        parser.add_argument(
            "--alg_palette_sam_redundancy_threshold",
            type=float,
            default=0.62,
            help="redundancy threshold above which redundant masks are not kept",
        )

        parser.add_argument(
            "--alg_palette_sam_sobel_threshold",
            type=float,
            default=0.7,
            help="sobel threshold in %% of gradient magintude",
        )

        parser.add_argument(
            "--alg_palette_sam_final_canny",
            action="store_true",
            default=False,
            help="whether to perform a Canny edge detection on sam sketch to soften the edges",
        )

        parser.add_argument(
            "--alg_palette_sam_min_mask_area",
            type=float,
            default=0.001,
            help="minimum area in proportion of image size for a mask to be kept",
        )

        parser.add_argument(
            "--alg_palette_sam_max_mask_area",
            type=float,
            default=0.99,
            help="maximum area in proportion of image size for a mask to be kept",
        )

        parser.add_argument(
            "--alg_palette_sam_points_per_side",
            type=int,
            default=16,
            help="number of points per side of image to prompt SAM with (# of prompted points will be points_per_side**2)",
        )

        parser.add_argument(
            "--alg_palette_sam_no_sample_points_in_ellipse",
            action="store_false",
            default=True,
            help="whether to not sample the points inside an ellipse to avoid the corners of the image",
        )

        parser.add_argument(
            "--alg_palette_sam_crop_delta",
            type=int,
            default=True,
            help="extend crop's width and height by 2*crop_delta before computing masks",
        )

        parser.add_argument(
            "--alg_palette_guidance_scale",
            type=float,
            default=0.0,  # literature value: 0.2
            help="scale for classifier-free guidance, default is conditional DDPM only",
        )

        parser.add_argument(
            "--min_crop_bbox_ratio",
            type=float,
            help="minimum crop/bbox ratio, allows to add context when bbox is larger than crop",
        )

        parser.add_argument(
            "--sampling-steps",
            type=int,
            default=0,
            help="nb of sampling steps for diffusion",
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

        parser.add_argument(
            "--sampling_method",
            type=str,
            default="ddpm",
            choices=["ddpm", "ddim"],
            help="choose the sampling method between ddpm and ddim",
        )

        parser.add_argument(
            "--ddim_num_steps",
            type=int,
            default=10,
            help="number of steps for ddim sampling method",
        )

        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.5,
            help="eta parameter for ddim variance",
        )

        parser.add_argument(
            "--cls-override",
            action="store_false",
            help="override input bbox classe for generation",
        )

        # TODO: fix and change to false in order to pass asset test
        self.isTrain = True
        self.isPredict = True
        return parser
