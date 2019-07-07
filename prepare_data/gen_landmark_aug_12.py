# coding: utf-8
import os
from os.path import exists

import cv2
import numpy as np

from prepare_data.utils import generateData





if __name__ == '__main__':
    dstdir = "../DATA/12/train_PNet_landmark_aug"
    output_path = '../DATA/12'
    data_path = '../DATA'
    if not exists(output_path):
        os.mkdir(output_path)
    if not exists(dstdir):
        os.mkdir(dstdir)
    assert (exists(dstdir) and exists(output_path))
    # train data
    net = "PNet"
    #the file contains the names of all the landmark training data
    train_txt = "prepare_data/trainImageList.txt"
    imgs,landmarks = generateData(train_txt, data_path, output_path, dstdir, net, argument=True)
    
   
