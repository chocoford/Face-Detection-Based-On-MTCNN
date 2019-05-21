#coding:utf-8
import sys, threading, queue, time
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np

class Detect_mode():
    simultaneous = 0
    concurrent = 1

detect_mode = Detect_mode.concurrent
test_mode = "onet"
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
#vis = True
detectors = [None, None, None]
prefix = ['data/MTCNN_model/PNet_landmark/PNet', 'data/MTCNN_model/RNet_landmark/RNet', 'data/MTCNN_model/ONet_landmark/ONet']
epoch = [30, 22, 22]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors[2] = ONet
mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 900)
video_capture.set(4, 1440)
corpbbox = None
boxes_c, landmarks = np.array([]), np.array([])

class Detect_thread(threading.Thread):
    def __init__(self, q, image):
        super(Detect_thread,self).__init__()
        self.image = image
        self.q = q
        self.t = 0
    def run(self):
        while True:
            if self.image is None or not self.q.empty():
                continue
            t1 = cv2.getTickCount()
            p = mtcnn_detector.detect(image)
            t2 = cv2.getTickCount()
            self.t = t2 - t1
            # if self.q.empty():
            self.q.put(p)
            

    def get_t(self):
        return self.t / cv2.getTickFrequency()

if detect_mode == Detect_mode.concurrent:
    result_queue = queue.Queue()
    detect_thread = Detect_thread(result_queue, None)
    detect_thread.start()



while True:
    ret, frame = video_capture.read()
    if ret:
        image = np.array(frame)
        
        if detect_mode == Detect_mode.concurrent:
            detect_thread.image = image
            if not result_queue.empty():
                boxes_c,landmarks = result_queue.get()
                
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            t = detect_thread.get_t()
            
        else:
            t1 = cv2.getTickCount()
            boxes_c,landmarks = mtcnn_detector.detect(image)
            t2 = cv2.getTickCount()
            t = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / t
        

        # print(landmarks.shape)
        
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # if score > thresh:
            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 158), 1)
        for i in range(landmarks.shape[0]):
            for j in range(landmarks.shape[1]//2):
                cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))            
        # time end
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('device not find')
        break
video_capture.release()
cv2.destroyAllWindows()
