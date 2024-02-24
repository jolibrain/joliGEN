import os.path
from data.unaligned_labeled_cls_dataset import UnalignedLabeledClsDataset
from data.base_dataset import BaseDataset
from data.online_creation import fill_mask_with_random, fill_mask_with_color
from data.image_folder import make_labeled_path_dataset
from data.sound_folder import load_sound
from PIL import Image
import numpy as np
import torch
from torch.fft import fft

# TODO optional?
import torchaudio
import warnings


class SelfSupervisedLabeledSoundDataset(UnalignedLabeledClsDataset):
    """
    This dataset class can create paired datasets with mask labels from only one domain.
    """

    def __init__(self, opt, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt, phase)

        self.A_img_paths, self.A_label = make_labeled_path_dataset(
            self.dir_A, "/paths.txt", opt.data_max_dataset_size
        )  # load images from '/path/to/data/trainA/paths.txt' as well as labels

        # Split multilabel
        self.A_label = [lbl.split(" ") for lbl in self.A_label]
        self.A_label = np.array(self.A_label, dtype=np.float32)

        self.A_size = len(self.A_img_paths)  # get the size of dataset A
        self.B_size = 0  # get the size of dataset B

    def get_img(
        self,
        A_sound_path,
        A_label_mask_path,
        A_label_cls,
        B_img_path=None,
        B_label_mask_path=None,
        B_label_cls=None,
        index=None,
    ):
        try:
            target = load_sound(A_sound_path)
            # XXX: some datasets don't convert to int, which mean they are never used with palette, because palette requires cls to be int
            A_label = torch.tensor(self.A_label[index % self.A_size].astype(int))
            result = {
                "A": torch.randn_like(target),
                "B": target,
                "A_img_paths": A_sound_path,
                "A_label_cls": A_label,
                "B_label_cls": A_label,
            }
        except Exception as e:
            print(e, "self supervised sound data loading")
            return None

        return result
