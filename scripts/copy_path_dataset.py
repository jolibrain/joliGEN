import os
import shutil
import argparse
import sys

parser = argparse.ArgumentParser(description='Copies a path-based dataset and all its data')
parser.add_argument('--path-file',help='input path file')
parser.add_argument('--dataroot-out-dir',help='output directory, subdirs will be img and bbox')
parser.add_argument('--domain-dir', help='output domain dir, must be one-level inside dataroot-out-dir')
args = parser.parse_args()

img_dir = args.dataroot_out_dir + '/' + args.domain_dir + '/img/'
bbox_dir = args.dataroot_out_dir + '/' + args.domain_dir + '/bbox/'
img_dir_rel = args.domain_dir + '/img/'
bbox_dir_rel = args.domain_dir + '/bbox/'

try:
    os.mkdir(img_dir)
    os.mkdir(bbox_dir)
except:
    print('failed creating directories, they may already exist')

out_fp = open(args.dataroot_out_dir + '/' + args.domain_dir + '/paths.txt','w+')

with open(args.path_file,'r') as fp:
    for line in fp:
        elts = line.split()
        img_path = elts[0]
        img_dest_path = img_dir + os.path.basename(img_path)
        bbox_path = elts[1]
        bbox_dest_path = bbox_dir + os.path.basename(bbox_path)

        succeed = True
        try:
            shutil.copyfile(img_path,img_dest_path)
        except:
            print('failed copying file ',img_path)
            succeed = False
            continue
        try:
            shutil.copyfile(bbox_path,bbox_dest_path)
        except:
            print('failed copying file ', bbox_path)
            succeed = False
            continue

        if succeed:
            img_dest_path_rel = img_dir_rel + os.path.basename(img_dest_path)
            bbox_dest_path_rel = bbox_dir_rel + os.path.basename(bbox_dest_path)
            out_fp.write(img_dest_path_rel + ' ' + bbox_dest_path_rel + '\n')

out_fp.close()
