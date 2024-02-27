from data.base_dataset import BaseDataset, get_transform
from data.utils import load_image
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data."""

    def __init__(self, opt, phase="train"):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt, phase)
        self.A_paths = sorted(make_dataset(self.dir_A, opt.data_max_dataset_size))
        self.transform = get_transform(opt, grayscale=(self.input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = load_image(A_path)
        A = self.transform(A_img)
        return {"A": A, "A_paths": A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
