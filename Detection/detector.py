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
    #rnet and onet minibatch(test)
    def predict(self, databatch):
        predictions = self.model.predict(databatch)
        cls_prob, bbox_pred, landmark_pred = predictions
        return cls_prob, bbox_pred, landmark_pred

        # # access data
        # # databatch: N x 3 x data_size x data_size
        # scores = []
        # batch_size = self.batch_size

        # minibatch = []
        # cur = 0
        # #num of all_data
        # n = databatch.shape[0]
        # while cur < n:
        #     #split mini-batch
        #     minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
        #     cur += batch_size
        # #every batch prediction result
        # cls_prob_list = []
        # bbox_pred_list = []
        # landmark_pred_list = []
        # for idx, data in enumerate(minibatch):
        #     m = data.shape[0]
        #     real_size = self.batch_size
        #     # the last batch 
        #     if m < batch_size:
        #         keep_inds = np.arange(m)
        #         #gap (difference)
        #         gap = self.batch_size - m
        #         while gap >= len(keep_inds):
        #             gap -= len(keep_inds)
        #             keep_inds = np.concatenate((keep_inds, keep_inds))
        #         if gap != 0:
        #             keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
        #         data = data[keep_inds]
        #         real_size = m
        #     #cls_prob batch*2
        #     #bbox_pred batch*4
        #     cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.image_op: data})
        #     #num_batch * batch_size *2
        #     cls_prob_list.append(cls_prob[:real_size])
        #     #num_batch * batch_size *4
        #     bbox_pred_list.append(bbox_pred[:real_size])
        #     #num_batch * batch_size*10
        #     landmark_pred_list.append(landmark_pred[:real_size])
        #     #num_of_data*2,num_of_data*4,num_of_data*10
        # return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)


