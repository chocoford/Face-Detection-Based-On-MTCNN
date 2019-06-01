import tensorflow as tf
import numpy as np
import os


class Detector(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self, net_factory, data_size, batch_size, model_path):

        self.model = net_factory()
        root = tf.train.Checkpoint(model=self.model)
        root.restore(tf.train.latest_checkpoint(model_path)).assert_existing_objects_matched()

        self.data_size = data_size
        self.batch_size = batch_size

    def predict(self, databatch):
        predictions = self.model.predict(databatch)
        cls_prob, bbox_pred, landmark_pred = predictions
        return cls_prob, bbox_pred, landmark_pred