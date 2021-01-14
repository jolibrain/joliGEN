from data.image_folder import make_labeled_mask_dataset
from data.unaligned_labeled_mask_dataset import UnalignedLabeledMaskDataset as UnalignedLabeledMaskDataset
from argparse import Namespace
from PIL import Image
from scipy import misc
import os
import os.path
import glob
import visdom
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
print('import ok')

def display_mask(mask):
    dict_col =np.array(
        [
            [0,0,0],
            [0,255,0],
            [255,0,0],
            [0,0,255],
            [255,255,255],
            [96,96,96],
            [253,96,96],
            [255,255,0],
            [237,127,16],
            [102,0,153],
        ]
    )

    dict_col =np.array(
        [
            [0,0,0], #black                                                                                                                                                                                        
            [0,255,0],#green                                                                                                                                                                                       
            [255,0,0],#red                                                                                                                                                                                         
            [0,0,255],#blue                                                                                                                                                                                        
            [0,255,255],#cyan                                                                                                                                                                                      
            [255,255,255],#white                                                                                                                                                                                   
            [96,96,96], #grey                                                                                                                                                                                      
            [255,255,0],#yellow                                                                                                                                                                                    
            [237,127,16],#orange                                                                                                                                                                                   
            [102,0,153],#purple                                                                                                                                                                                    
            [88,41,0], #brown                                                                                                                                                                                      
            [253,108,158],#pink                                                                                                                                                                                    
            [128,0,0],#maroon                                                                                                                                                                                      
            [255,0,255],
            [255,0,127],
            [0,128,255],
            [0,102,51],#17                                                                                                                                                                                         
            [192,192,192],
            [128,128,0],
            [84, 151, 120]

        ]
    )

    
    try :
        len(mask.shape)==2
    except AssertionError:
        print('Mask\'s shape is not 2')
    print('mask shape',mask.shape)
    mask_dis = np.zeros((mask.shape[0],mask.shape[1],3))
    dic =[]
    for i in mask:
        for j in i:
            if j not in dic:
                dic.append(j)
    print('dic',dic)
    print('mask_dis shape',mask_dis.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
#            print(i,j)
#            print('(mask_dis[i,j,:].shape',mask_dis[i,j,:].shape)
#            print('mask[i,j]',mask[i,j])
#            print('dict_col[mask[i,j]].shape',dict_col[mask[i,j]].shape)
            mask_dis[i,j,:] = dict_col[mask[i,j]]
    print('ok')
    return mask_dis

vis = visdom.Visdom(port='1235')

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        #print(image_numpy.shape)
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        #print('len',len(image_numpy.shape))
        if len(image_numpy.shape)!=2:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        else :
            print('MASK')
            print('shape',image_numpy.shape)
            image_numpy = image_numpy.astype(np.uint8)
            image_numpy = display_mask(image_numpy)
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


dir = '/data1/pnsuau/cartier/markers2ring'
#dir = '/data1/pnsuau/planes/cropped_centered'
#paths = 'test.txt'

#print(make_labeled_mask_dataset_2(dir,paths))

#opt {'phase','dataroot','max_dataset_size'}

opt =Namespace(batch_size=128, beta1=0.5, checkpoints_dir='./checkpoints', continue_train=False, dataroot=dir, dataset_mode='unaligned_labeled_mask_2', direction='AtoB', display_env='main', display_freq=400, dispay_id=1, display_ncols=4, display_port=8097, display_server='http://localhost', display_winsize=256, epoch='latest', epoch_count=1, gan_mode='lsgan', gpu_ids=[2], init_gain=0.02, init_type='normal', input_nc=3, isTrain=True, lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5, load_iter=0, load_size=480, lr=0.0002, lr_decay_iters=50, lr_policy='linear', max_dataset_size='inf', model='cycle_gan_semantic', n_layers_D=3, name='svhn2mnist_test_1', ndf=64, netD='basic', netG='resnet_9blocks', ngf=64, niter=100, niter_decay=100, no_dropout=False, no_flip=True, no_html=False, norm='instance', num_threads=4, output_nc=3, phase='train', pool_size=50, preprocess='resize_and_crop', print_freq=100, save_by_iter=False, save_epoch_freq=5, save_latest_freq=5000, semantic_nclasses=10, serial_batches=False, suffix='', update_html_freq=1000, verbose=False,crop_size=480)

print(opt.beta1)

dataset = UnalignedLabeledMaskDataset(opt)

#for i in range (len(dataset)):    
for i, data in enumerate(dataset):
    data = dataset.__getitem__(i)
    #print(data)
    im =np.transpose(tensor2im(data['A']),(2,0,1))
    #print('----------------------------------------------',im.shape)
    #temp = np.copy(im[0])
    #im[0]=np.copy(im[2])
    #im[2]=np.copy(temp)
    vis.image(im)
    
    #print('A',data['A'])
    #print('label',data['A_label'])
    #print('label',data['A_label'].shape)
    #im =  np.transpose(tensor2im(data['A_label'][0]),(2,0,1))
    im = tensor2im(data['A_label'][0])
    print('----------------------------------------------label',im.shape)
    im = np.array(im,dtype=np.uint8)
    im = np.transpose(im,(2,0,1))
    dic = []
    #temp = np.copy(im[0])
    #im[0]=np.copy(im[2])
    #im[2]=np.copy(temp)
    vis.image(im)
    im =np.transpose(tensor2im(data['B']),(2,0,1))
    #temp = np.copy(im[0])
    #im[0]=np.copy(im[2])
    #im[2]=np.copy(temp)
    vis.image(im)
print(len(dataset))



im = cv2.imread('/data1/pnsuau/cityscapes/resized_480x480_8cls/train/imgs/bayreuth_000000_000313_leftImg8bit.png')
#print(im)
#im = np.array(im)
im = np.transpose(im, (2,0,1))
#print(im)
#print('shape objectif',im.shape)
#vis.image(im)

im_lab = cv2.imread('/data1/pnsuau/cityscapes/resized_480x480_8cls/train/annot/berlin_000170_000019_gtFine_labelTrainIds.png')
#print(im_lab)
#im_lab = np.array(im_lab,dtype=np.uint8)
dict=[]
#for k in im_lab[0]:
#    for i in k :
#        if i not in dict:
#            dict.append(i)
#print(dict)

#print('im label',im_lab)
#print(im_lab.shape)
im_lab_dis = display_mask(im_lab)

#print(len(np.array([0]).shape))

#print('im label dis  shape',im_lab_dis.shape)
#print(np.max(im_lab))
#im_lab = np.transpose(im_lab, (2,0,1))
#print(im_lab_dis)

#vis.image(im_lab_dis)



dict = []
for k in range (1):
    print(k)
    data_1 = dataset.__getitem__(0)
    print(data_1['A'].shape)
    im = tensor2im(data_1['A'])
    print('A',im.shape)
    im = np.transpose(im,(2,0,1))
    print('lab',im.shape)
    vis.image(im) #il faut inverser les channels et images np.transpose......

    im = tensor2im(data_1['B'])
    print('B',im.shape)
    im = np.transpose(im,(2,0,1))
    print('lab',im.shape)
    vis.image(im) #il faut inverser les channels et images np.transpose......
    
    
    im =  np.transpose(tensor2im(data_1['A_label']),(2,0,1))
    print(im.shape)
    im = display_mask(im)
    vis.image(im)
    

transform_list = []
transform_list += [transforms.ToTensor()]
transforms.Compose(transform_list)


dir = '/data1/pnsuau/cityscapes/resized_480x480_8cls/train/train_sample.txt'

#with  open(dir, 'r') as f:
#    paths_list = f.read().split('\n')
#for line in paths_list:
 #   line_split = line.split(' ')
#    if len(line_split)==2:
#        im = Image.open(line_split[0])
#        im = np.array(im)
#        print(im.shape)
#        im = np.transpose(im, (2,0,1))
#        vis.image(im)

#        im = Image.open(line_split[1])
#        im = np.array(im,dtype=np.uint8)
#        print(im.shape)
#        im = display_mask(im)
#        vis.image(im)
