# coding: utf-8
import os
from os.path import exists
from prepare_data.utils import generateData

if __name__ == '__main__':
    dstdir = "../DATA/48/train_ONet_landmark_aug"
    OUTPUT = '../DATA/48'
    if not exists(OUTPUT): os.mkdir(OUTPUT)
    if not exists(dstdir): os.mkdir(dstdir)
    assert(exists(dstdir) and exists(OUTPUT))

    # train data
    net = "ONet"
    #train_txt = "train.txt"
    train_txt = "prepare_data/trainImageList.txt"
    data_dir = "../DATA"
    imgs,landmarks = generateData(train_txt, data_dir, OUTPUT, dstdir, net, argument=True)
    #WriteToTfrecord(imgs,landmarks,net)
   
