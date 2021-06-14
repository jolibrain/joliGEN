import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_list
from data.image_folder import make_dataset, make_labeled_mask_dataset, make_dataset_path,sort_nicely
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as F
import glob

class NupletUnalignedLabeledMaskDataset(BaseDataset):
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

        self.nuplet_size = self.opt.nuplet_size
        
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        if os.path.exists(self.dir_A):
            self.A_img_paths, self.A_label_paths = make_labeled_mask_dataset(self.dir_A,'/paths.txt', opt.max_dataset_size)   # load images from '/path/to/data/trainA/paths.txt' as well as labels
        else:
            self.A_img_paths, self.A_label_paths = make_labeled_mask_dataset(opt.dataroot,'/paths.txt', opt.max_dataset_size)   # load images from '/path/to/data/trainA/paths.txt' as well as labels
        self.A_size = len(self.A_img_paths)  # get the size of dataset A

        if os.path.exists(self.dir_B):
            self.B_img_paths, self.B_label_paths = make_labeled_mask_dataset(self.dir_B,'/paths.txt', opt.max_dataset_size)    # load images from '/path/to/data/trainB'
            self.B_size = len(self.B_img_paths)  # get the size of dataset B

        self.transform=get_transform_list(self.opt, grayscale=(self.input_nc == 1))

        
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
        A_list=[]
        A_label_list=[]
        A_paths_list=[]
        
        index_A = index

        A_img_path = self.A_img_paths[index_A % (self.A_size- self.nuplet_size)]  # make sure index is within then range
        A_label_path = self.A_label_paths[index_A % (self.A_size- self.nuplet_size)]
        
        A_paths_list=glob.glob(A_img_path+"/*")
        A_label_paths_list=glob.glob(A_label_path+"/*")

        sort_nicely(A_paths_list)
        sort_nicely(A_label_list)

        for cur_A_img_path,cur_A_label_path in zip(A_paths_list,A_label_paths_list):
            try:
                A_img = Image.open(cur_A_img_path).convert('RGB')
                A_label = Image.open(cur_A_label_path)
            except Exception as e:
                print('failure with reading A domain image ', A_img_path, ' or label ', A_label_path)
                print(e)
       
            #A,A_label = self.transform(A_img,A_label)

            A_list.append(A_img)
            A_label_list.append(A_label)


        A_list,A_label_list = self.transform(A_list,A_label_list)

        A = torch.stack(A_list)
        A_label = torch.stack(A_label_list)
            
        B_list=[]
        B_label_list=[]
        B_paths_list=[]
        if hasattr(self,'B_img_paths') :
            if self.opt.serial_batches:   # make sure index is within then range
                index_B = index % (self.B_size - self.nuplet_size)
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1 - self.nuplet_size)

            B_img_path = self.B_img_paths[index_B]
            B_paths_list=glob.glob(B_img_path+"/*")
            sort_nicely(B_paths_list)
            if len(self.B_label_paths) > 0: # B label is optional
                B_label_path = self.B_label_paths[index_B]
                B_label_paths_list=glob.glob(B_label_path+"/*")
                sort_nicely(B_label_paths_list)
            else:
                B_label_list = [None for k in range(len(B_img_path))]
                
            for cur_B_img_path,cur_B_label_path in zip(B_paths_list,B_label_paths_list):
                try:
                    B_img = Image.open(cur_B_img_path).convert('RGB')
                    if cur_B_label_path is not None: # B label is optional
                        B_label = Image.open(cur_B_label_path)
                    else:
                        B_label = None
                except:
                    print("failed to read B domain image ", B_img_path, " at index_B=", index_B)
                        
                if len(self.B_label_paths) > 0: # B label is optional
                    #B,B_label = self.transform(B_img,B_label)
                    B_label_list.append(B_label)
                else:
                    #B = self.transform_noseg(B_img)
                    B_label_list = None
                    
                    
                B_list.append(B_img)

            B_list,B_label_list = self.transform(B_list,B_label_list)
            B = torch.stack(B_list)
            if len(self.B_label_paths) > 0: # B label is optional
                B_label = torch.stack(B_label_list)
            
            return {'A': A, 'B': B, 'A_paths': A_img_path, 'B_paths': B_img_path, 'A_label': A_label, 'B_label': B_label}
        else:
            return {'A': A, 'A_paths': A_img_path,'A_label': A_label}


    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if hasattr(self,'B_img_paths'):
            return max(self.A_size, self.B_size)
        else:
            return self.A_size
