import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import  make_labeled_mask_dataset, make_dataset_path
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms
import torch

class UnalignedLabeledMask2Dataset(BaseDataset):
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
        
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_img_paths, self.A_label_paths = make_labeled_mask_dataset(self.dir_A,'/paths.txt', opt.max_dataset_size)   # load images from '/path/to/data/trainA/paths.txt' as well as labels
        self.B_img_paths = make_dataset_path(self.dir_B,'/paths.txt', opt.max_dataset_size)    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_img_paths)  # get the size of dataset A
        self.B_size = len(self.B_img_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_label = get_transform(self.opt,convert=False,method=Image.NEAREST)
        
                
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
            A_label (tensor) -- mask label of image A
        """
    
        A_img_path = self.A_img_paths[index % self.A_size]  # make sure index is within then range
        A_label_path = self.A_label_paths[index % self.A_size]
        
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
            
        B_img_path = self.B_img_paths[index_B]
#        B_label_path = self.B_label_paths[index_B]# % self.B_size]
        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        
        # get labels
        transform_list = [transforms.ToTensor()]
        toten = transforms.Compose(transform_list)

        A_label = self.transform_label(Image.open(A_label_path))#pn:[index  % self.A_size] ?
#        B_label = self.transform_label(Image.open(B_label_path))#pn:[index  % self.A_size] ?

        im_np = np.array(A_label,dtype=np.int64)
        im_np = np.array([im_np],dtype=np.int64)
        A_label = torch.from_numpy(im_np)

#        im_np = np.array(B_label,dtype=np.int64)
#        im_np = np.array([im_np],dtype=np.int64)
#        B_label = torch.from_numpy(im_np)
        
        return {'A': A, 'B': B, 'A_paths': A_img_path, 'B_paths': B_img_path, 'A_label': A_label}#, 'B_label': B_label}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
