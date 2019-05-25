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
import tensorflow as tf

tf.enable_eager_execution()

test_mode = "PNet"
thresh = [0.6, 0.7, 0.7]
min_face_size = 20
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['../data/ultramodern_model/PNet', '../data/ultramodern_model/RNet', '../data/ultramodern_model/ONet']
batch_size = [2048, 64, 16]
# model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
# load pnet model
PNet = FCNDetector(P_Net, prefix[0])
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], prefix[1])
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], prefix[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
test_img_path = []
path = "test/test images"
for item in os.listdir(path):
    test_img_path.append(os.path.join(path,item))
# test_data = TestLoader(gt_imdb)


all_boxes,landmarks = mtcnn_detector.detect_face(test_img_path)

count = 0
for imagepath in test_img_path:
    print(imagepath)
    image = cv2.imread(imagepath)
    for bbox in all_boxes[count]:
        # if bbox[4] < 0.9:
        #     continue
        cv2.putText(image, str(np.round(bbox[4],2)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_TRIPLEX, 1, color=(255,0,255))
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])),(0,0,255), 1)
        if bbox[0] < 0:
            print("there is a bbox[0] < 0", bbox[0])
    # for landmark in landmarks[count]:
    #     for i in range(landmark.shape[0]//2):
    #         cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))
            
    count = count + 1
    cv2.namedWindow("test result",0)
    cv2.imshow("test result",image)
    cv2.waitKey(0)