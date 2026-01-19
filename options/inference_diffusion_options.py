from options import BaseOptions
from util.util import MAX_INT, pairs_of_floats, pairs_of_ints


class InferenceDiffusionOptions(BaseOptions):
    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        parser = super().initialize(parser)

        parser.add_argument("--name", help="inference process name", default="predict")

        parser.add_argument(
            "--model_in_file",
            help="file path to generator model (.pth file)",
            required=True,
        )

        parser.add_argument(
            "--paths_in_file",
            help="frames and bbox pairwise file path for the video generation(.txt file)",
            required=False,
        )
        parser.add_argument("--img_in", help="image to transform", required=True)
        parser.add_argument(
            "--dir_out",
            help="The directory where to output result images",
            required=True,
        )

        parser.add_argument(
            "--img_width",
            default=-1,
            type=int,
            help="image width, defaults to model crop size",
        )
        parser.add_argument(
            "--img_height",
            default=-1,
            type=int,
            help="image height, defaults to model crop size",
        )
        parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
        parser.add_argument("--gpuid", type=int, default=0, help="which GPU to use")

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
        parser.add_argument(
            "--seed", type=int, default=-1, help="random seed for reproducibility"
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
            "--cls",
            type=int,
            default=-1,
            help="override input bbox classe for generation",
        )

        # XXX: options that are not in gen_single_video
        parser.add_argument("--previous_frame", help="image to transform", default=None)
        parser.add_argument(
            "--mask_in", help="mask used for image transformation", required=False
        )
        parser.add_argument("--ref_in", help="image used as reference", required=False)
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
            "--min_crop_bbox_ratio",
            type=float,
            help="minimum crop/bbox ratio, allows to add context when bbox is larger than crop",
        )

        # XXX: options that are also in palette_model
        parser.add_argument(
            "--alg_diffusion_cond_image_creation",
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
                "pix2pix",
            ],
            help="how cond_image is created",
        )
        parser.add_argument(
            "--alg_diffusion_guidance_scale",
            type=float,
            default=0.0,  # literature value: 0.2
            help="scale for classifier-free guidance, default is conditional DDPM only",
        )
        parser.add_argument(
            "--alg_diffusion_sketch_canny_thresholds",
            type=int,
            nargs="+",
            default=[0, 255 * 3],
            help="Canny thresholds",
        )
        parser.add_argument(
            "--alg_diffusion_super_resolution_downsample",
            action="store_true",
            help="whether to downsample the image for super resolution",
        )
        parser.add_argument(
            "--alg_diffusion_sam_use_gaussian_filter",
            action="store_true",
            default=False,
            help="whether to apply a gaussian blur to each SAM masks",
        )

        parser.add_argument(
            "--alg_diffusion_sam_no_sobel_filter",
            action="store_false",
            default=True,
            help="whether to not use a Sobel filter on each SAM masks",
        )

        parser.add_argument(
            "--alg_diffusion_sam_no_output_binary_sam",
            action="store_false",
            default=True,
            help="whether to not output binary sketch before Canny",
        )

        parser.add_argument(
            "--alg_diffusion_sam_redundancy_threshold",
            type=float,
            default=0.62,
            help="redundancy threshold above which redundant masks are not kept",
        )

        parser.add_argument(
            "--alg_diffusion_sam_sobel_threshold",
            type=float,
            default=0.7,
            help="sobel threshold in %% of gradient magnitude",
        )

        parser.add_argument(
            "--alg_diffusion_sam_final_canny",
            action="store_true",
            default=False,
            help="whether to perform a Canny edge detection on sam sketch to soften the edges",
        )

        parser.add_argument(
            "--alg_diffusion_sam_min_mask_area",
            type=float,
            default=0.001,
            help="minimum area in proportion of image size for a mask to be kept",
        )

        parser.add_argument(
            "--alg_diffusion_sam_max_mask_area",
            type=float,
            default=0.99,
            help="maximum area in proportion of image size for a mask to be kept",
        )

        parser.add_argument(
            "--alg_diffusion_sam_points_per_side",
            type=int,
            default=16,
            help="number of points per side of image to prompt SAM with (# of prompted points will be points_per_side**2)",
        )

        parser.add_argument(
            "--alg_diffusion_sam_no_sample_points_in_ellipse",
            action="store_false",
            default=True,
            help="whether to not sample the points inside an ellipse to avoid the corners of the image",
        )

        parser.add_argument(
            "--alg_diffusion_sam_crop_delta",
            type=int,
            default=True,
            help="extend crop's width and height by 2*crop_delta before computing masks",
        )

        parser.add_argument(
            "--model_prior_321_backwardcompatibility",
            action="store_true",
            help="whether to load models from previous version of JG.",
        )

        parser.add_argument(
            "--alg_palette_ddim_num_steps",
            type=int,
            default=10,
            help="number of steps for ddim sampling method",
        )

        parser.add_argument(
            "--alg_palette_ddim_eta",
            type=float,
            default=0.5,
            help="eta parameter for ddim variance",
        )

        parser.add_argument(
            "--f_s_weight_sam",
            type=str,
            default="models/configs/sam/pretrain/sam_vit_b_01ec64.pth",
            help="path to sam weight for f_s, e.g. models/configs/sam/pretrain/sam_vit_b_01ec64.pth",
        )
        parser.add_argument(
            "--data_refined_mask",
            action="store_true",
            help="whether to use refined mask with sam",
        )
        parser.add_argument(
            "--vid_frame_extension_number",
            type=int,
            default=0,
            help="additional sequence number added during the inference",
        )
        parser.add_argument(
            "--alg_sc_denoise_inferstep",
            type=int,
            default=1,
            help="denoise steps during the inference time",
        )

        parser.add_argument(
            "--alg_b2b_denoise_inferstep",
            type=int,
            default=1,
            help="denoise steps during the inference time",
        )

        parser.add_argument(
            "--vid_fps",
            type=float,
            default=18,
            help="fps during the inference video generation",
        )

        self.isTrain = False
        return parser
