import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_seg
from data.image_folder import make_dataset, make_labeled_path_dataset, make_dataset_path
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as F

class UnalignedLabeledMaskDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets with mask labels.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    
    Domain A must have labels, at the moment there are two subdirections 'images' and 'labels'.
    
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        if os.path.exists(self.dir_A):
            self.A_img_paths, self.A_label_paths = make_labeled_path_dataset(self.dir_A,'/paths.txt', opt.max_dataset_size)   # load images from '/path/to/data/trainA/paths.txt' as well as labels
        else:
            self.A_img_paths, self.A_label_paths = make_labeled_path_dataset(opt.dataroot,'/paths.txt', opt.max_dataset_size)   # load images from '/path/to/data/trainA/paths.txt' as well as labels
        self.A_size = len(self.A_img_paths)  # get the size of dataset A

        if os.path.exists(self.dir_B):
            self.B_img_paths, self.B_label_paths = make_labeled_path_dataset(self.dir_B,'/paths.txt', opt.max_dataset_size)    # load images from '/path/to/data/trainB'
            self.B_size = len(self.B_img_paths)  # get the size of dataset B

        self.transform=get_transform_seg(self.opt, grayscale=(self.input_nc == 1))
        self.transform_noseg=get_transform(self.opt, grayscale=(self.input_nc == 1))

    def get_img(self,A_img_path,A_label_path,B_img_path=None,B_label_path=None,index=None):
        try:
            A_img = Image.open(A_img_path).convert('RGB')
            A_label = Image.open(A_label_path)
        except Exception as e:
            print('failure with reading A domain image ', A_img_path, ' or label ', A_label_path)
            print(e)
            return None
       
        A,A_label = self.transform(A_img,A_label)
        if self.opt.all_classes_as_one:
            A_label = (A_label >= 1)*1

        if B_img_path is not None:
            try:
                B_img = Image.open(B_img_path).convert('RGB')
                if B_label_path is not None:
                    B_label = Image.open(B_label_path)
                    B,B_label = self.transform(B_img,B_label)
                else:
                    B = self.transform_noseg(B_img)
                    B_label = []
            except:
                print("failed to read B domain image ", B_img_path, " or label", B_label_path)
                return None
            
            if len(self.B_label_paths) > 0: # B label is optional
                B_label_path = self.B_label_paths[index_B]
                B_label = Image.open(B_label_path)
                B,B_label = self.transform(B_img,B_label)
                if self.opt.all_classes_as_one:
                    B_label = (B_label >= 1)*1

            else:
                B = self.transform_noseg(B_img)
                B_label = []
        
            return {'A': A, 'B': B, 'A_img_paths': A_img_path, 'B_img_paths': B_img_path, 'A_label': A_label, 'B_label': B_label}

        else:
            return {'A': A, 'A_img_paths': A_img_path,'A_label': A_label}
        
    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if hasattr(self,'B_img_paths'):
            return max(self.A_size, self.B_size)
        else:
            return self.A_size
