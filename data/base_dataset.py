"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""

import random
import os.path
import numpy as np
import torch.utils.data as data
from PIL import Image

import torch
import torchvision
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms

if torch.__version__[0] == "2":
    torchvision.disable_beta_transforms_warning()
    from torchvision import tv_tensors as datapoints
    from torchvision.transforms.v2 import functional as F2

import torchvision.transforms.functional as F

from data.image_folder import make_dataset, make_dataset_path, make_labeled_path_dataset
from data.online_creation import sanitize_paths, write_paths_file

from abc import ABC, abstractmethod
import imgaug as ia
import imgaug.augmenters as iaa
import os
import warnings


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt, phase, name=""):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
            phase (str)       -- can be train,test or validation.
        """
        self.phase = phase
        self.name = name
        self.opt = opt

        self.use_domain_B = not "self_supervised" in self.opt.data_dataset_mode

        self.root = opt.dataroot

        if not self.root.endswith("/"):
            self.root += "/"

        self.sv_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.warning_mode = self.opt.warning_mode
        self.set_dataset_dirs_and_dims()

        if opt.data_image_bits > 8 and opt.model_input_nc > 1:
            self.use_tiff = True  # multi-channel images > 8bit
        else:
            self.use_tiff = False

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @staticmethod
    def modify_commandline_options_train(parser):
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
            A_label_mask (tensor) -- mask label of image A
        """
        if not self.warning_mode:
            warnings.simplefilter("ignore")

        A_img_path = self.A_img_paths[
            index % self.A_size
        ]  # make sure index is within then range
        if hasattr(self, "A_label_mask_paths") and len(self.A_label_mask_paths) > 0:
            A_label_mask_path = self.A_label_mask_paths[index % self.A_size]
        else:
            A_label_mask_path = None

        if hasattr(self, "A_label_cls"):
            A_label_cls = self.A_label_cls[index % self.A_size]
        else:
            A_label_cls = None

        if hasattr(self, "B_img_paths"):
            if self.opt.data_serial_batches:  # make sure index is within then range
                index_B = index % self.B_size
            else:  # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)

            B_img_path = self.B_img_paths[index_B]

            if (
                hasattr(self, "B_label_mask_paths") and len(self.B_label_mask_paths) > 0
            ):  # B label is optional
                B_label_mask_path = self.B_label_mask_paths[index_B]
            else:
                B_label_mask_path = None

            if (
                hasattr(self, "B_label_cls") and len(self.B_label_cls) > 0
            ):  # B label is optional
                B_label_cls = self.B_label_cls[index_B]
            else:
                B_label_cls = None
        else:
            B_img_path = None
            B_label_mask_path = None
            B_label_cls = None

        if self.opt.data_relative_paths:
            A_img_path = os.path.join(self.root, A_img_path)
            if A_label_mask_path is not None:
                A_label_mask_path = os.path.join(self.root, A_label_mask_path)

            if hasattr(self, "B_img_paths"):
                B_img_path = os.path.join(self.root, B_img_path)
                if B_label_mask_path is not None:
                    B_label_mask_path = os.path.join(self.root, B_label_mask_path)

        results = self.get_img(
            A_img_path,
            A_label_mask_path,
            A_label_cls,
            B_img_path,
            B_label_mask_path,
            B_label_cls,
            index,
        )

        return results

    def set_dataset_dirs_and_dims(self):
        btoA = self.opt.data_direction == "BtoA"
        name = self.name if self.phase == "test" else ""
        self.input_nc = (
            self.opt.model_output_nc if btoA else self.opt.model_input_nc
        )  # get the number of channels of input image
        self.output_nc = (
            self.opt.model_input_nc if btoA else self.opt.model_output_nc
        )  # get the number of channels of output image

        if not btoA:
            self.dir_A = os.path.join(
                self.opt.dataroot, self.phase + "A" + name
            )  # create a path '/path/to/data/trainA'

            if self.use_domain_B:
                self.dir_B = os.path.join(
                    self.opt.dataroot, self.phase + "B" + name
                )  # create a path '/path/to/data/trainB'
        else:
            self.dir_A = os.path.join(
                self.opt.dataroot, self.phase + "B" + name
            )  # create a path '/path/to/data/trainB'

            if self.use_domain_B:
                self.dir_B = os.path.join(
                    self.opt.dataroot, self.phase + "A" + name
                )  # create a path '/path/to/data/trainA'

    def get_validation_set(self, size):
        return_A_list = []
        return_B_list = []
        if not hasattr(self, "A_label_mask_paths_val"):
            A_label_mask_paths_val = [None for k in range(size)]
        else:
            A_label_mask_paths_val = self.A_label_mask_paths_val

        if not hasattr(self, "A_label_cls_val"):
            A_label_cls_val = [None for k in range(size)]
        else:
            A_label_cls_val = self.A_label_cls_val

        if not hasattr(self, "B_img_paths_val"):
            self.B_img_paths_val = [None for k in range(size)]
        if not hasattr(self, "B_label_mask_paths_val"):
            B_label_mask_paths_val = [None for k in range(size)]
        else:
            B_label_mask_paths_val = self.B_label_mask_paths_val

        if not hasattr(self, "B_label_cls_val"):
            B_label_cls_val = [None for k in range(size)]
        else:
            B_label_cls_val = self.B_label_cls_val

        for index, (
            A_img_path,
            A_label_mask_path,
            A_label_cls,
            B_img_path,
            B_label_mask_path,
            B_label_cls,
        ) in enumerate(
            zip(
                self.A_img_paths_val,
                A_label_mask_paths_val,
                A_label_cls_val,
                self.B_img_paths_val,
                B_label_mask_paths_val,
                B_label_cls_val,
            )
        ):
            if len(return_A_list) >= size:
                break

            if self.opt.data_relative_paths:
                A_img_path = os.path.join(self.root, A_img_path)
                if A_label_mask_path is not None:
                    A_label_mask_path = os.path.join(self.root, A_label_mask_path)
                B_img_path = os.path.join(self.root, B_img_path)
                if B_label_mask_path is not None:
                    B_label_mask_path = os.path.join(self.root, B_label_mask_path)

            images = self.get_img(
                A_img_path,
                A_label_mask_path,
                A_label_cls,
                B_img_path,
                B_label_mask_path,
                B_label_cls,
                index,
            )
            if images is not None:
                return_A_list.append(images["A"].unsqueeze(0))
                if "B" in images:
                    return_B_list.append(images["B"].unsqueeze(0))

        return_A_list = torch.cat(return_A_list)
        if return_B_list[0] is not None:
            return_B_list = torch.cat(return_B_list)

        return return_A_list, return_B_list

    def sanitize(self):
        paths_sanitized_train_A = os.path.join(
            self.sv_dir, "paths_sanitized_train_A.txt"
        )
        if hasattr(self, "B_img_paths"):
            paths_sanitized_train_B = os.path.join(
                self.sv_dir, "paths_sanitized_train_B.txt"
            )
        if hasattr(self, "B_img_paths"):
            train_sanitized_exist = os.path.exists(
                paths_sanitized_train_A
            ) and os.path.exists(paths_sanitized_train_B)
        else:
            train_sanitized_exist = os.path.exists(paths_sanitized_train_A)

        if train_sanitized_exist:
            self.A_img_paths, self.A_label_mask_paths = make_labeled_path_dataset(
                self.sv_dir, "/paths_sanitized_train_A.txt"
            )
            if hasattr(self, "B_img_paths"):
                self.B_img_paths, self.B_label_mask_paths = make_labeled_path_dataset(
                    self.sv_dir, "/paths_sanitized_train_B.txt"
                )
        else:
            print("--------------")
            print("Sanitizing images and labels paths")
            print("--- DOMAIN A ---")

            self.A_img_paths, self.A_label_mask_paths = sanitize_paths(
                self.A_img_paths,
                self.A_label_mask_paths,
                mask_delta=self.opt.data_online_creation_mask_delta_A,
                mask_random_offset=self.opt.data_online_creation_mask_random_offset_A,
                crop_delta=self.opt.data_online_creation_crop_delta_A,
                mask_square=self.opt.data_online_creation_mask_square_A,
                crop_dim=self.opt.data_online_creation_crop_size_A,
                output_dim=self.opt.data_load_size,
                max_dataset_size=self.opt.data_max_dataset_size,
                context_pixels=self.opt.data_online_context_pixels,
                load_size=self.opt.data_online_creation_load_size_A,
                select_cat=self.opt.data_online_select_category,
                data_relative_paths=self.opt.data_relative_paths,
                data_root_path=self.opt.dataroot,
            )
            write_paths_file(
                self.A_img_paths,
                self.A_label_mask_paths,
                paths_sanitized_train_A,
            )

            print("--- DOMAIN B ---")
            if hasattr(self, "B_img_paths"):
                self.B_img_paths, self.B_label_mask_paths = sanitize_paths(
                    self.B_img_paths,
                    self.B_label_mask_paths,
                    mask_delta=self.opt.data_online_creation_mask_delta_B,
                    mask_random_offset=self.opt.data_online_creation_mask_random_offset_B,
                    crop_delta=self.opt.data_online_creation_crop_delta_B,
                    mask_square=self.opt.data_online_creation_mask_square_B,
                    crop_dim=self.opt.data_online_creation_crop_size_B,
                    output_dim=self.opt.data_load_size,
                    max_dataset_size=self.opt.data_max_dataset_size,
                    context_pixels=self.opt.data_online_context_pixels,
                    load_size=self.opt.data_online_creation_load_size_B,
                    data_relative_paths=self.opt.data_relative_paths,
                    data_root_path=self.opt.dataroot,
                )
                write_paths_file(
                    self.B_img_paths,
                    self.B_label_mask_paths,
                    paths_sanitized_train_B,
                )

            print("--------------")


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.data_preprocess == "resize_and_crop":
        new_h = new_w = opt.data_load_size
    elif opt.data_preprocess == "scale_width_and_crop":
        new_w = opt.data_load_size
        new_h = opt.data_load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.data_crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.data_crop_size))

    fliph = random.random() > 0.5
    flipv = random.random() > 0.5
    angle = random.randint(0, 3) * 90  # 0, 90, 180, 270

    return {"crop_pos": (x, y), "fliph": fliph, "flipv": flipv, "angle": angle}


def get_transform(
    opt,
    params=None,
    grayscale=False,
    method=InterpolationMode.BICUBIC,
    convert=True,
    crop=True,
):
    margin = opt.data_online_context_pixels * 2

    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if "resize" in opt.data_preprocess:
        osize = [opt.data_load_size + margin, opt.data_load_size + margin]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif "scale_width" in opt.data_preprocess:
        transform_list.append(
            transforms.Lambda(
                lambda img: __scale_width(img, opt.data_load_size, method)
            )
        )

    if "crop" in opt.data_preprocess and crop:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.data_crop_size + margin))
        else:
            transform_list.append(
                transforms.Lambda(
                    lambda img: __crop(
                        img, params["crop_pos"], opt.data_crop_size + margin
                    )
                )
            )

    # if opt.data_preprocess == "none":
    #   transform_list.append(
    #       transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method))
    #   )

    if opt.dataaug_flip != "none":
        if params is None:
            if opt.dataaug_flip == "horizontal":
                transform_list.append(transforms.RandomHorizontalFlip())  # default
            elif opt.dataaug_flip == "vertical":
                transform_list.append(transforms.RandomVerticalFlip())
            elif opt.dataaug_flip == "both":
                transform_list.append(transforms.RandomHorizontalFlip())
                transform_list.append(transforms.RandomVerticalFlip())
        else:
            if params["fliph"]:
                transform_list.append(
                    transforms.Lambda(
                        lambda img: __flip(img, params["fliph"], horizontal=True)
                    )
                )
            if params["flipv"]:
                transform_list.append(
                    transforms.Lambda(
                        lambda img: __flip(img, params["flipv"], horizontal=False)
                    )
                )

    if not opt.dataaug_no_rotate:
        if params is None or not "angle" in params:
            transform_list.append(transforms.RandomRotation([-90, 180]))
        elif params["angle"]:
            transform_list.append(
                transforms.Lambda(lambda img: __rotate(img, params["angle"]))
            )

    if opt.dataaug_affine:
        transform_list.append(
            transforms.RandomAffine(
                0,
                (opt.dataaug_affine_translate, opt.dataaug_affine_translate),
                (opt.dataaug_affine_scale_min, opt.dataaug_affine_scale_max),
                (-opt.dataaug_affine_shear, opt.dataaug_affine_shear),
            )
        )

    if opt.dataaug_imgaug and not grayscale:
        transform_list.append(RandomImgAug(with_mask=False))

    if convert:
        transform_list += [transforms.ToTensor()]  # if not uint8, no scaling
        if opt.data_image_bits > 8:
            transform_list += [torchvision.transforms.v2.ToDtype(torch.float32)]
            bit_scaling = 2**opt.data_image_bits - 1
            transform_list += [
                transforms.Lambda(lambda img: img * (1 / float(bit_scaling)))
            ]
            transform_list += [
                transforms.Normalize((0.5,), (0.5,))
            ]  # XXX: > 8bit, mono canal only for now
        else:
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
    return transforms.Compose(transform_list)


def get_transform_ref(
    opt,
    params=None,
    grayscale=False,
    method=InterpolationMode.BICUBIC,
    convert=True,
    crop=True,
):
    transform_list = []

    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    osize = [opt.data_crop_size, opt.data_crop_size]
    transform_list.append(transforms.Resize(osize, interpolation=method))

    if convert:
        transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=InterpolationMode.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), interpolation=method)


def __scale_width(img, target_width, method=InterpolationMode.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), interpolation=method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip, horizontal=True):
    if flip:
        if horizontal:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def __rotate(img, angle):
    if angle != 0:
        return img.rotate(angle)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print(
            "The image size needs to be a multiple of 4. "
            "The loaded image size was (%d, %d), so it was adjusted to "
            "(%d, %d). This adjustment will be done to all images "
            "whose sizes are not multiples of 4" % (ow, oh, w, h)
        )
        __print_size_warning.has_printed = True


def get_transform_seg(
    opt, params=None, grayscale=False, method=InterpolationMode.BICUBIC
):
    margin = opt.data_online_context_pixels * 2

    transform_list = []
    print("method seg", method)

    if grayscale:
        transform_list.append(GrayscaleMask(1))

    if "resize" in opt.data_preprocess:
        osize = [
            opt.data_load_size + margin,
            opt.data_load_size + margin,
        ]
        transform_list.append(ResizeMask(osize, interpolation=method))

    if "crop" in opt.data_preprocess:
        transform_list.append(RandomCropMask(opt.data_crop_size + margin))

    if opt.dataaug_imgaug:
        if not grayscale:
            transform_list.append(RandomImgAug())

    if opt.dataaug_flip != "none":
        if opt.dataaug_flip == "horizontal":
            transform_list.append(RandomHorizontalFlipMask())
        elif opt.dataaug_flip == "vertical":
            transform_list.append(RandomVerticalFlipMask())
        elif opt.dataaug_flip == "both":
            transform_list.append(RandomHorizontalFlipMask())
            transform_list.append(RandomVerticalFlipMask())
        transform_list.append(RandomHorizontalFlipMask())

    if not opt.dataaug_no_rotate:
        transform_list.append(
            RandomRotationMask(degrees=0)
        )  # XXX: degrees is a required placeholder, unused

    if opt.dataaug_affine:
        raff = RandomAffineMask(degrees=0)
        raff.set_params(
            opt.dataaug_affine,
            opt.dataaug_affine_translate,
            opt.dataaug_affine_scale_min,
            opt.dataaug_affine_scale_max,
            opt.dataaug_affine_shear,
        )
        transform_list.append(raff)

    transform_list += [ToTensorMask()]

    if grayscale:
        transform_list += [NormalizeMask((0.5,), (0.5,))]
    else:
        transform_list += [NormalizeMask((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return ComposeMask(transform_list)


class ComposeMask(transforms.Compose):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __call__(self, img, mask, bbox=None):
        if bbox is None:
            w, h = img.size
            bbox = np.array([0, 0, w, h])  # sets bbox to full image size
        if torch.__version__[0] == "2":
            tbbox = datapoints.BoundingBoxes(
                bbox, format=datapoints.BoundingBoxFormat.XYXY, canvas_size=img.size
            )
        else:
            tbbox = bbox  # placeholder
        for t in self.transforms:
            img, mask, tbbox = t(img, mask, tbbox)
        return img, mask, tbbox


class GrayscaleMask(transforms.Grayscale):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img, mask, bbox):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return (
            F.to_grayscale(img, num_output_channels=self.num_output_channels),
            mask,
            bbox,
        )

    def __repr__(self):
        return self.__class__.__name__ + "(num_output_channels={0})".format(
            self.num_output_channels
        )


class ResizeMask(transforms.Resize):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __call__(self, img, mask, bbox):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        if torch.__version__[0] == "2":
            return (
                F.resize(img, self.size, interpolation=self.interpolation),
                F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST),
                F2.resize(bbox, self.size),
            )
        else:
            return (
                F.resize(img, self.size, interpolation=self.interpolation),
                F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST),
                [],
            )


class RandomCropMask(transforms.RandomCrop):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __call__(self, img, mask, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(
                img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode
            )
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(
                img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode
            )

        i, j, h, w = self.get_params(img, self.size)

        if torch.__version__[0] == "2":
            return (
                F.crop(img, i, j, h, w),
                F.crop(mask, i, j, h, w),
                F2.crop(bbox, i, j, h, w),
            )
        else:
            return (
                F.crop(img, i, j, h, w),
                F.crop(mask, i, j, h, w),
                [],
            )


class RandomHorizontalFlipMask(transforms.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __call__(self, img, mask, bbox):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            if torch.__version__[0] == "2":
                return F.hflip(img), F.hflip(mask), F2.hflip(bbox)
            else:
                return F.hflip(img), F.hflip(mask), []
        return img, mask, bbox


class RandomVerticalFlipMask(transforms.RandomVerticalFlip):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __call__(self, img, mask, bbox):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            if torch.__version__[0] == "2":
                return F.vflip(img), F.vflip(mask), F2.vflip(bbox)
            else:
                return F.vflip(img), F.vflip(mask), []
        return img, mask, bbox


class ToTensorMask(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, img, mask, bbox):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if torch.__version__[0] == "2":
            bbdata = bbox.data
        else:
            bbdata = bbox
        return (
            F.to_tensor(img),
            torch.from_numpy(np.array(mask, dtype=np.int64)).unsqueeze(0),
            bbdata,
        )


class RandomRotationMask(transforms.RandomRotation):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, mask, bbox):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        angle = random.choice([0, 90, 180, 270])

        if torch.__version__[0] == "2":
            return (
                F.rotate(img, angle),
                F.rotate(mask, angle, fill=(0,)),
                F2.rotate(bbox, angle),
            )
        else:
            return (
                F.rotate(img, angle),
                F.rotate(mask, angle, fill=(0,)),
                [],
            )


class NormalizeMask(transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __call__(self, tensor_img, tensor_mask, tensor_bbox):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return (
            F.normalize(tensor_img, self.mean, self.std, self.inplace),
            tensor_mask,
            tensor_bbox,
        )

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class RandomAffineMask(transforms.RandomAffine):
    """Apply random affine transform"""

    def set_params(self, p, translate, scale_min, scale_max, shear):
        self.p = p
        self.translate = translate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shear = shear

    def __call__(self, img, mask, bbox):
        if random.random() > 1.0 - self.p:
            affine_params = self.get_params(
                (0, 0),
                (self.translate, self.translate),
                (self.scale_min, self.scale_max),
                (-self.shear, self.shear),
                img.size,
            )
            if torch.__version__[0] == "2":
                return (
                    F.affine(img, *affine_params),
                    F.affine(mask, *affine_params),
                    F2.affine(bbox, *affine_params),
                )
            else:
                return (
                    F.affine(img, *affine_params),
                    F.affine(mask, *affine_params),
                    [],
                )
        else:
            return img, mask, bbox


def sometimes(aug):
    return iaa.Sometimes(0.5, aug)


class RandomImgAug:
    def __init__(self, with_mask=True):
        self.with_mask = with_mask
        self.seq = iaa.Sequential(
            [
                iaa.SomeOf(
                    (0, 5),
                    [
                        sometimes(
                            iaa.Superpixels(p_replace=(0, 0.5), n_segments=(100, 200))
                        ),  # convert images into their superpixel representation
                        iaa.OneOf(
                            [
                                iaa.GaussianBlur(
                                    (0, 3.0)
                                ),  # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(
                                    k=(2, 7)
                                ),  # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(
                                    k=(3, 11)
                                ),  # blur image using local medians with kernel sizes between 2 and 7
                            ]
                        ),
                        iaa.Sharpen(
                            alpha=(0, 1.0), lightness=(0.75, 1.5)
                        ),  # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        iaa.SimplexNoiseAlpha(
                            iaa.OneOf(
                                [
                                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                    iaa.DirectedEdgeDetect(
                                        alpha=(0.5, 1.0), direction=(0.0, 1.0)
                                    ),
                                ]
                            )
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),  # add gaussian noise to images
                        # iaa.OneOf([
                        #    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        #    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        # ]),
                        iaa.Invert(0.05, per_channel=True),  # invert color channels
                        iaa.Add(
                            (-5, 5), per_channel=0.5
                        ),  # change brightness of images
                        iaa.AddToHueAndSaturation(
                            (-20, 20)
                        ),  # change hue and saturation
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.OneOf(
                            [
                                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                iaa.FrequencyNoiseAlpha(
                                    exponent=(-4, 0),
                                    first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                    second=iaa.LinearContrast((0.5, 2.0)),
                                ),
                            ]
                        ),
                        iaa.LinearContrast(
                            (0.5, 2.0), per_channel=0.5
                        ),  # improve or worsen the contrast
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                    ],
                    random_order=True,
                )
            ],
            random_order=True,
        )

    def __call__(self, img, mask, bbox):
        tarr = self.seq(image=np.array(img))
        nimg = Image.fromarray(tarr)
        if self.with_mask:
            return nimg, mask, bbox
        else:
            return nimg


################################################################


def get_transform_list(
    opt, params=None, grayscale=False, method=InterpolationMode.BICUBIC
):
    margin = opt.data_online_context_pixels * 2

    transform_list = []
    print("method seg", method)

    if grayscale:
        transform_list.append(GrayscaleMaskList(1))

    if "resize" in opt.data_preprocess:
        osize = [
            opt.data_load_size + margin,
            opt.data_load_size + margin,
        ]
        transform_list.append(ResizeMaskList(osize, interpolation=method))

    if "crop" in opt.data_preprocess:
        transform_list.append(RandomCropMaskList(opt.data_crop_size + margin))

    if opt.dataaug_flip != "none":
        if opt.dataaug_flip == "horizontal":
            transform_list.append(RandomHorizontalFlipMaskList())
        elif opt.dataaug_flip == "vertical":
            transform_list.append(RandomVerticalFlipMaskList())
        elif opt.dataaug_flip == "both":
            transform_list.append(RandomHorizonalFlipMaskList())
            transform_list.append(RandomVerticalFlipMaskList())

    if not opt.dataaug_no_rotate:
        transform_list.append(RandomRotationMaskList(degrees=0))

    if opt.dataaug_affine:
        raff = RandomAffineMaskList(degrees=0)
        raff.set_params(
            opt.dataaug_affine,
            opt.dataaug_affine_translate,
            opt.dataaug_affine_scale_min,
            opt.dataaug_affine_scale_max,
            opt.dataaug_affine_shear,
        )
        transform_list.append(raff)

    transform_list += [ToTensorMaskList()]

    if grayscale:
        transform_list += [NormalizeMaskList((0.5,), (0.5,))]
    else:
        transform_list += [NormalizeMaskList((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return ComposeMaskList(transform_list)


class ComposeMaskList(transforms.Compose):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __call__(self, imgs, masks=None, bbox=None):
        if bbox is None:
            w, h = imgs[0].size
            bbox = np.array([0, 0, w, h])  # sets bbox to full image size
        if torch.__version__[0] == "2":
            tbbox = datapoints.BoundingBoxes(
                bbox,
                format=datapoints.BoundingBoxFormat.XYXY,
                canvas_size=imgs[0].size,
            )
        else:
            tbbox = bbox  # placeholder
        for t in self.transforms:
            imgs, masks, tbbox = t(imgs, masks, tbbox)
        return imgs, masks, tbbox


class GrayscaleMaskList(transforms.Grayscale):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, imgs, masks, bbox):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return_imgs = []
        for img in imgs:
            return_imgs.append(
                F.to_grayscale(img, num_output_channels=self.num_output_channels)
            )

        return return_imgs, masks, bbox

    def __repr__(self):
        return self.__class__.__name__ + "(num_output_channels={0})".format(
            self.num_output_channels
        )


class ResizeMaskList(transforms.Resize):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __call__(self, imgs, masks, bbox):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """

        return_imgs = []
        return_masks = []

        for img in imgs:
            return_imgs.append(
                F.resize(img, self.size, interpolation=self.interpolation)
            )
        if masks is None:
            return masks
        else:
            for mask in masks:
                return_masks.append(
                    F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
                )
            return return_imgs, return_masks, F2.resize(bbox, self.size)


class RandomCropMaskList(transforms.RandomCrop):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __call__(self, imgs, masks, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return_imgs, return_masks = [], []
        for img in imgs:
            if self.padding is not None:
                img = F.pad(img, self.padding, self.fill, self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(
                    img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode
                )
            # pad the height if needed
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(
                    img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode
                )

            i, j, h, w = self.get_params(img, self.size)

            return_imgs.append(F.crop(img, i, j, h, w))
        if masks is None:
            return_masks = None
        else:
            for mask in masks:
                return_masks.append(F.crop(mask, i, j, h, w))
        return return_imgs, return_masks, F2.crop(bbox, i, j, h, w)


class RandomHorizontalFlipMaskList(transforms.RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __call__(self, imgs, masks, bbox):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return_imgs, return_masks = [], []
            for img in imgs:
                return_imgs.append(F.hflip(img))
            if masks is not None:
                for mask in masks:
                    return_masks.append(F.hflip(mask))
            else:
                return_masks = None

            return return_imgs, return_masks, F2.hflip(bbox)
        else:
            return imgs, masks, bbox


class RandomVerticalFlipMaskList(transforms.RandomVerticalFlip):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __call__(self, imgs, masks, bbox):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return_imgs, return_masks = [], []
            for img in imgs:
                return_imgs.append(F.vflip(img))
            if masks is not None:
                for mask in masks:
                    return_masks.append(F.vflip(mask))
            else:
                return_masks = None

            return return_imgs, return_masks, F2.vflip(bbox)
        else:
            return imgs, masks, bbox


class ToTensorMaskList(transforms.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, imgs, masks, bbox):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return_imgs, return_masks = [], []
        for img in imgs:
            return_imgs.append(F.to_tensor(img))
        if masks is not None:
            for mask in masks:
                return_masks.append(
                    torch.from_numpy(np.array(mask, dtype=np.int64)).unsqueeze(0)
                )
        else:
            return_masks = None
        return return_imgs, return_masks, bbox.data


class RandomRotationMaskList(transforms.RandomRotation):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, imgs, masks, bbox):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        angle = random.choice([0, 90, 180, 270])

        return_imgs, return_masks = [], []
        for img in imgs:
            return_imgs.append(F.rotate(img, angle))
        if masks is not None:
            for mask in masks:
                return_masks.append(F.rotate(mask, angle, fill=(0,)))
        else:
            return_masks = None

        return return_imgs, return_masks, F2.rotate(bbox, angle)


class NormalizeMaskList(transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __call__(self, tensor_imgs, tensor_masks, tensor_bbox):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """

        return_imgs, return_masks = [], []
        for tensor_img in tensor_imgs:
            return_imgs.append(
                F.normalize(tensor_img, self.mean, self.std, self.inplace)
            )

        return return_imgs, tensor_masks, tensor_bbox

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class RandomAffineMaskList(transforms.RandomAffine):
    """Apply random affine transform"""

    def set_params(self, p, translate, scale_min, scale_max, shear):
        self.p = p
        self.translate = translate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.shear = shear

    def __call__(self, imgs, masks, bbox):
        if random.random() > 1.0 - self.p:
            affine_params = self.get_params(
                (0, 0),
                (self.translate, self.translate),
                (self.scale_min, self.scale_max),
                (-self.shear, self.shear),
                img.size,
            )
            return_imgs, return_masks = [], []
            for img in imgs:
                return_imgs.append(F.affine(img, *affine_params))
            if masks is not None:
                for mask in masks:
                    return_masks.append(F.affine(mask, *affine_params))
            else:
                return_masks = None

            return return_imgs, return_masks, F2.affine(bbox, *affine_params)
        else:
            return imgs, masks, bbox
