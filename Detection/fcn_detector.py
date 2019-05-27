import tensorflow as tf
import numpy as np
import sys, os
sys.path.append("../")
from train_models.MTCNN_config import config
from train_models.mtcnn_model import P_Net

from train_models.utils import load_and_get_normalization_img

class FCNDetector(object):
    """
        Dectector for Fully Convolution Network.  
        用于全卷积网络的检测器
    """
    def __init__(self, net_factory, model_path):
        self.model = net_factory()
        root = tf.train.Checkpoint(model=self.model)
        root.restore(tf.train.latest_checkpoint(model_path)).assert_existing_objects_matched()
        # cls_pred = self.model(tf.expand_dims(load_and_get_normalization_img("test/not test/2333.jpg"), axis=0))[0]
        # cls_numpy = cls_pred.numpy()
        # print("raw prediction: ", cls_numpy)
        # print("233")

    def predict(self, databatch):
        databatch = np.expand_dims(databatch, axis=0)
        databatch = tf.cast(databatch, tf.float32)
        pred = self.model(databatch)
        cls_prob, bbox_pred, _ = pred
        assert(cls_prob.shape[3]==2 and bbox_pred.shape[3]==4)
        cls_prob = np.squeeze(cls_prob, axis=0)
        bbox_pred = np.squeeze(bbox_pred, axis=0)
        return cls_prob, bbox_pred
