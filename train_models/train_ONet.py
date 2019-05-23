#coding:utf-8
from train_models.mtcnn_model import O_Net, cls_ohem, bbox_ohem, landmark_ohem, cal_accuracy
from train_models.utils import get_dataset
import tensorflow as tf
import time, os, sys
tf.enable_eager_execution()

def cls_loss(model, x, label):
    cls_prob = model(x)[0]
    return cls_ohem(cls_prob,label)

def bbox_loss(model, x, bbox_target, label):
    bbox_pred = model(x)[1]
    return bbox_ohem(bbox_pred,bbox_target,label)

def landmark_loss(model, x,landmark_target, label):
    landmark_pred = model(x)[2]
    return landmark_ohem(landmark_pred, landmark_target, label)


def cls_acc(cls_pred, labels):
    return cal_accuracy(cls_pred, labels)


def loss(model, images, labels, bboxes, landmarks):
    c_loss = cls_loss(model, images, labels)
    b_loss = bbox_loss(model, images, bboxes, labels)
    l_loss = landmark_loss(model, images, landmarks, labels)
    return c_loss + 0.5 * b_loss + l_loss

def grad(model, images, labels, bboxes, landmarks):
    with tf.GradientTape() as tape:
        loss_value = loss(model, images, labels, bboxes, landmarks)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_ONet(base_dir, checkpoint_dir, end_epoch, display_step, lr):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    model = O_Net()
    batch_size = 256
    total_num, train_dataset = get_dataset(base_dir, batch_size=batch_size)
    optimizer = tf.train.MomentumOptimizer(lr, 0.9)


    # prepare for save
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    root = tf.train.Checkpoint(optimizer=optimizer, model=model)

    now = time.time()
    pred = now

    print("start training")
    for epoch in range(end_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.Mean()

        for i, train_batch in enumerate(train_dataset):
            images, target_batch = train_batch
            labels, bboxes, landmarks = target_batch
            loss_value, grads = grad(model, images, labels, bboxes, landmarks)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            acc_value = cls_acc(model(images)[0], labels)

            epoch_loss_avg(loss_value)
            epoch_accuracy_avg(acc_value)

            if i % display_step == 0:
                now = time.time()
                total_steps = total_num // batch_size
                remaining_time = (now - pred) * (total_steps - i) / display_step // 60
                sys.stdout.write("\r>> {0} of {1} steps done. Estimated remaining time: {2} mins. loss_value: {3} acc: {4}".format(i, 
                                                                                                                                   total_steps, 
                                                                                                                                   remaining_time,
                                                                                                                                   loss_value.numpy(),
                                                                                                                                   acc_value.numpy()))
                sys.stdout.flush()  
                pred = now

        print("\rEpoch {0}: Loss: {1} Accuracy: {2} ".format(epoch, epoch_loss_avg.result(), epoch_accuracy_avg.result()))

        # save model
        save_path = root.save(checkpoint_prefix)
        print("save prefix is {}".format(save_path))





if __name__ == '__main__':

    base_dir = '../DATA/imglists/ONet'
    model_name = 'ultramodern'
    model_path = '../data/%s_model/ONet_landmark/ONet' % model_name
    prefix = model_path
    end_epoch = 22
    display = 10
    lr = 0.001
    train_ONet(base_dir, prefix, end_epoch, display, lr)
