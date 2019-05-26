#coding:utf-8
import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
from prepare_data.utils import load_wider_face_gt_boxes, IoU
import cv2
import os
import numpy as np
thresh = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2
shuffle = False
detectors = [None, None, None]
prefix = ['data/MTCNN_model/PNet_landmark/PNet', 'data/MTCNN_model/RNet_landmark/RNet', 'data/MTCNN_model/ONet_landmark/ONet']
epoch = [30, 22, 22]
batch_size = [2048, 64, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]

PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh)
gt_imdb = []
path = "E:/Document/Datasets/Wider Face/WIDER_val/images"

gt_data = load_wider_face_gt_boxes("wider_face_val_bbx_gt.txt")

for item in gt_data.keys():
    gt_imdb.append(os.path.join(path,item))
test_data = TestLoader(gt_imdb)


all_boxes,landmarks = mtcnn_detector.detect_face(test_data)

count = 0
scores = []
recall_rate = 0

for imagepath in gt_imdb:
    for bbox in all_boxes[count]:

        rate = len(all_boxes)

        score = 0
        for gt_boxes in gt_data[imagepath]:
            iou = IoU(bbox, gt_boxes)
            if score > iou:
                score = iou
        
    count = count + 1
