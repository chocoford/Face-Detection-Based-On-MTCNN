#coding:utf-8
import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FCNDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import os
import numpy as np
test_mode = "ONet"
thresh = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2
detectors = [None, None, None]
prefix = ['data/MTCNN_model/PNet_landmark/PNet', 'data/MTCNN_model/RNet_landmark/RNet', 'data/MTCNN_model/ONet_landmark/ONet']
# load pnet model
# PNet = FcnDetector(P_Net, model_path[0])
# detectors[0] = PNet

# # load rnet model
# if test_mode in ["RNet", "ONet"]:
#     RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
#     detectors[1] = RNet

# # load onet model
# if test_mode == "ONet":
#     ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
#     detectors[2] = ONet

# mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
#                                stride=stride, threshold=thresh, slide_window=slide_window)



#load FDDB datasets
def load_FDDB_dataset(base_dir):
    ellipseLists_path = []
    for i in range(10):
        path = "FDDB-fold-0{}-ellipseList.txt".format(i+1)
        path = os.path.join(base_dir, path)
        ellipseLists_path.append(path)

    gt_data = {}

    for path in ellipseLists_path:
        f = open(path)
        data = f.read()
        f.close()

        lines = data.split('\n')

        i = 0
        while True:
            gt_box_num = 1 if int(lines[i+1]) == 0 else int(lines[i+1])
            if int(lines[i+1]) == 0:
                print("no face: ", i+1)
            gt_pos = np.zeros((gt_box_num, 4))
            for j, bbox_list in enumerate([x.split(' ')[:5] for x in lines[i+2 : i+gt_box_num+2]]):
                gt_pos[j, 0] = float(bbox_list[3]) - float(bbox_list[1]) + 1.
                gt_pos[j, 1] = float(bbox_list[4]) - float(bbox_list[0]) + 1.
                gt_pos[j, 2] = float(bbox_list[3]) + float(bbox_list[1]) - 1.
                gt_pos[j, 3] = float(bbox_list[4]) + float(bbox_list[0]) - 1.
            gt_data[os.path.join(base_dir, lines[i])] = gt_pos
            i += gt_box_num + 2
            if i >= len(lines) - 1: #最后一行有一个换行
                break
    
    return gt_data


gt_data = load_FDDB_dataset("E:/Document/Datasets/FDDB")
gt_data

paths = gt_data.keys()
all_boxes, _ = mtcnn_detector.detect_face(paths)