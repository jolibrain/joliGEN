import sys
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.utils import load_image
from data.image_folder import make_dataset
from PIL import Image
import tifffile


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directories '/path/to/data/trainA' and '/path/to/data/trainB' contain image pairs with the same names.
    During test time, you need to prepare directories '/path/to/data/testA' and '/path/to/data/testB'.
    """

    def __init__(self, opt, phase, name=""):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt, phase, name)

        self.A_paths = sorted(
            make_dataset(self.dir_A, opt.data_max_dataset_size)
        )  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(
            make_dataset(self.dir_B, opt.data_max_dataset_size)
        )  # load images from '/path/to/data/trainB'

        if len(self.A_paths) != len(self.B_paths):
            raise Exception(
                "aligned dataset: domain A and domain B should have the same number of images"
            )

        if opt.data_image_bits > 8 and opt.model_input_nc > 1:
            self.use_tiff = True  # multi-channel images > 8bit
        else:
            self.use_tiff = False

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a pair of images given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]

        if self.use_tiff:
            A = tifffile.imread(A_path)
            B = tifffile.imread(B_path)
        else:
            A = Image.open(A_path)
            B = Image.open(B_path)

        if self.opt.data_image_bits == 8:
            A = A.convert("RGB")
            B = B.convert("RGB")
            grayscale = self.input_nc == 1
        else:  # for > 8 bit, no explicit conversion
            grayscale = False

        # apply the same transform to both A and B
        if self.use_tiff:
            transform_params = get_params(self.opt, A.shape[:2])
        else:
            transform_params = get_params(self.opt, A.size)

        A_transform = get_transform(self.opt, transform_params, grayscale=grayscale)
        B_transform = get_transform(self.opt, transform_params, grayscale=grayscale)

        # resize B to A's size with PIL
        if not self.use_tiff and A.size != B.size:
            B = B.resize(A.size, Image.NEAREST)

        A = A_transform(A)
        B = B_transform(B)

        return {"A": A, "B": B, "A_img_paths": A_path, "B_img_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
