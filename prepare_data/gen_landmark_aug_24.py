# coding: utf-8
import os
from os.path import exists
from prepare_data.utils import generateData



if __name__ == '__main__':
    dstdir = "../DATA/24/train_RNet_landmark_aug"
    OUTPUT = '../DATA/24'
    if not exists(OUTPUT): os.mkdir(OUTPUT)
    if not exists(dstdir): os.mkdir(dstdir)
    assert(exists(dstdir) and exists(OUTPUT))

    # train data
    net = "RNet"
    #train_txt = "train.txt"
    train_txt = "prepare_data/trainImageList.txt"
    data_dir = "data"
    imgs,landmarks = generateData(train_txt, data_dir, OUTPUT, dstdir, net, argument=True)
    
   
