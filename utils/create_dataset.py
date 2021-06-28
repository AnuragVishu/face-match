"""
    Dataset preparation file for CelebA dataset
    CelebA dataset consists of:
    1) An image directory consisting of all faces: img_align_celeba
    2) An identity text file consists of image-class mapping: identity_CelebA.txt
        Test file format is as follows
        <image1_name> <class1>
        <image2_name> <class2>
        <image3_name> <class2>
        <image4_name> <class1>
        ...
        <image10002_name> <class5000>
        ...

    This script prepare dataset for resnet training.
    The script prepare data in the following format:

        img_align_celeba_final
        |
        |---<class1>
                |
                |---<image1>
                |---<image4>
        |
        |---<class2>
                |
                |--<image2>
                |--<image3>
        ...
"""     

import os
from glob import glob
import cv2
import argparse
import sys
#import shutils

parser = argparse.ArgumentParser('Parse dataset constants')
parser.add_argument('--download', action="store_true", help='Download CelebA dataset')
parser.add_argument('--image', metavar='image', type=str, help='Path to image directory', required='--download' not in sys.argv)
parser.add_argument('--meta', metavar='meta', type=str, help='Path to Identity text file', required='--download' not in sys.argv)
parser.add_argument('--output', metavar='output', type=str, help='Path to output directory', required='--download' not in sys.argv)

args = parser.parse_args()
data = args.image
export = args.output
identity_file = args.meta

def stdprint(*args, **kwargs):
    returnval = kwargs['returnn']
    del kwargs['returnn']
    print(*args, **kwargs)

    return returnval

def make_directory(dir):
    print(f'Creating directory {dir}')
    try:
        os.mkdir(dir)
        return True
    except Exception as e:
        print(f'Error creating directory: {dir}')

# data locations
# data = 'img_align_celeba'

# open identity file
file = open(identity_file, 'r')
lines = file.readlines()

# create write directory
st = make_directory(export) if not os.path.isdir(export) else stdprint(f'directory {export} already exists', returnn=False)

count = 0
for line in lines:
    count+=1
    print(line)
    
    # extract metadata    
    filename, category = line.split(" ")

    # read file from data pool
    print(f'Reading file {filename} category {category}')
    im = cv2.imread(os.path.join(data, filename))
    
    # copy files to respective dir
    fullpath = os.path.join(export, str(category))
    st = make_directory(fullpath) if not os.path.isdir(fullpath) else stdprint(f'directory {fullpath} already exists', returnn=False)
    filefullpath=os.path.join(fullpath, filename)
  
    print(f'copying {filename} to {filefullpath}')
    try:
        cv2.imwrite(filefullpath, im)
    except Exception as e:
        print(f'error writing file {filename} to {fullpath} : {e}')
