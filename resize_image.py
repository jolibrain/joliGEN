from PIL import Image
from scipy import misc
import os
import os.path
import glob
import visdom
import numpy as np
#from util import util, html

vis =visdom.Visdom(port='1234')


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

dir = '/media/data/datasets/images/gan/gta5/images'

k=0
all_files = glob.glob(dir + '/*')
for img in all_files:
    if is_image_file(img) and k<100:
        #print('path',img)
        name = img.split('/')[-1]
        print('name',name)
        k+=1
        im = Image.open(img)
        #print(im.size)
        box=((im.size[0]-im.size[1])/2,0,im.size[1]+(im.size[0]-im.size[1])/2,im.size[1])
        #im_crop =im.copy().crop(box)
        im_resize =im.copy().crop(box)
        im_resize = im_resize.resize((480,480))
        im_resize_np = np.array(im_resize)
        im_resize_np = np.transpose(im_resize_np, (2,0,1))
        #vis.image(im_np)
        im_resize.save('/data1/pnsuau/gta5/gta5_sample_480x480/'+ name)
        print('ok')

dir_2 ='/data1/pnsuau/gta5/gta5_sample_480x480'

all_files = glob.glob(dir_2 + '/*')
k=0
for img in all_files:
    if is_image_file(img):
        print('Image',k)
        im = Image.open(img)
        im = np.array(im)
        im = np.transpose(im, (2,0,1))
        vis.image(im)
        k+=1
        
