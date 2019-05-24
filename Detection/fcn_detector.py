import tensorflow as tf
import numpy as np
import sys, os
sys.path.append("../")
from train_models.MTCNN_config import config
from train_models.mtcnn_model import P_Net

class FCNDetector(object):
    """
        Dectector for Fully Convolution Network.  
        用于全卷积网络的检测器
    """
    def __init__(self, net_factory, model_path):

        self.model = P_Net()
        # optimizer = tf.train.MomentumOptimizer(0.001, 0.9)
        checkpoint_dir = model_path
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        root = tf.train.Checkpoint(model=self.model)
        root.restore(tf.train.latest_checkpoint(checkpoint_prefix))


    def predict(self, databatch):
        # height, width, _ = databatch.shape
        # print(height, width)
        databatch = np.expand_dims(databatch, axis=0)
        pred = self.model.predict(databatch)
        cls_prob, bbox_pred, _ = pred
        assert(cls_prob.shape[3]==2 and bbox_pred.shape[3]==4)
        cls_prob = np.squeeze(cls_prob, axis=0)
        bbox_pred = np.squeeze(bbox_pred, axis=0)
        return cls_prob, bbox_pred


class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, net_factory, model_path):
        #create a graph
        graph = tf.Graph()
        with graph.as_default():
            #define tensor and op in graph(-1,1)
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            #self.cls_prob batch*2
            #self.bbox_pred batch*4
            #construct model here
            #self.cls_prob, self.bbox_pred = net_factory(image_reshape, training=False)
            #contains landmark
            self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)
            
            #allow 
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print("restore models' param")
            saver.restore(self.sess, model_path)
    def predict(self, databatch):
        height, width, _ = databatch.shape
        # print(height, width)
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                                           feed_dict={self.image_op: databatch, self.width_op: width,
                                                                      self.height_op: height})
        return cls_prob, bbox_pred
