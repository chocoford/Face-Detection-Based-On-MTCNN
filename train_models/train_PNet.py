#coding:utf-8
from train_models.mtcnn_model import P_Net, cls_ohem, bbox_ohem, landmark_ohem, cal_accuracy
from train_models.train import train
import tensorflow as tf
import os, sys, time
import random
from prepare_data.read_tfrecord_v2 import read_multi_tfrecords,read_single_tfrecord
from train_models.utils import get_dataset
tf.enable_eager_execution()


def cls_loss(model, x, label):
    cls_prob = tf.squeeze(model(x)[0], [1, 2])
    return cls_ohem(cls_prob,label)

def bbox_loss(model, x, bbox_target, label):
    bbox_pred = tf.squeeze(model(x)[1], [1, 2])
    return bbox_ohem(bbox_pred,bbox_target,label)

def landmark_loss(model, x,landmark_target, label):
    landmark_pred = tf.squeeze(model(x)[2], [1, 2])
    return landmark_ohem(landmark_pred, landmark_target, label)

def multi_loss(pred, target):
    print(target)
    print(pred)
    # label, bbox_target, landmark_target = target
    # cls_pred, bbox_pred, landmark_pred = pred

    # cls_pred = tf.squeeze(cls_pred, [1, 2])
    # bbox_pred = tf.squeeze(bbox_pred, [1, 2])
    # landmark_pred = tf.squeeze(landmark_pred, [1, 2])

    # cls_loss = cls_ohem(cls_pred,label)
    # bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
    # landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)

    return 0.5

def cls_acc(cls_pred, labels):
    cls_pred = tf.squeeze(cls_pred, [1, 2])
    return cal_accuracy(cls_pred, labels)


def loss(model, images, labels, bboxes, landmarks):
    c_loss = cls_loss(model, images, labels)
    b_loss = bbox_loss(model, images, bboxes, labels)
    l_loss = landmark_loss(model, images, landmarks, labels)
    return c_loss + 0.5 * b_loss + 0.5 * l_loss

def grad(model, images, labels, bboxes, landmarks):
    with tf.GradientTape() as tape:
        loss_value = loss(model, images, labels, bboxes, landmarks)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)



def train_PNet(base_dir, prefix, end_epoch, display, lr):
    """
        train PNet
    Parameters:
    ---------------
        base_dir:
        dataset_dir: tfrecord路径
        prefix: model路径
        end_epoch: 训练最大轮次数
        display: 每训练display个step输出训练状态
        lr: 学习率
    """
    model = P_Net()
    batch_size = 256
    total_num, train_dataset = get_dataset("../data/imglists/PNet", batch_size=batch_size)

    # callbacks = [tf.keras.callbacks.ModelCheckpoint("../data/ultramodern_model/PNet/pnet.h5",
    #                                                 monitor="multi_loss", 
    #                                                 save_best_only=True),
                                                    
    #             ]

    optimizer = tf.train.MomentumOptimizer(lr, 0.9)

    # 计算损失时会用到额外数据，所以只能自己写training loop
    # losses = {
    #     "cls_output": "categorical_crossentropy",
    #     "bbox_output": "categorical_crossentropy",
    # }
    # lossWeights = {"category_output": 1.0, "color_output": 1.0}

    # model.compile(optimizer, loss=multi_loss, metrics=[cls_acc])
    # model.fit(train_dataset, epochs=30, steps_per_epoch=total_num//batch_size, callbacks=callbacks)


    display_step = 100

    #estimate time left
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

        print("\rEpoch {0}: Loss: {1} Accuracy: {2}".format(epoch, epoch_loss_avg.result(), epoch_accuracy_avg.result()))

        # save model
        checkpoint_dir = prefix
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        root = tf.train.Checkpoint(optimizer=optimizer, model=model)
        save_path = root.save(checkpoint_prefix)
        print("save prefix is {}".format(save_path))

if __name__ == '__main__':
    #data path
    base_dir = '../DATA/imglists/PNet'
    model_name = 'ultramodern'
    #with landmark
    model_path = '../data/%s_model/PNet' % model_name
            
    prefix = model_path
    end_epoch = 30
    display = 100
    lr = 0.001
    train_PNet(base_dir, prefix, end_epoch, display, lr)
