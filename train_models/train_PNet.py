#coding:utf-8
from train_models.mtcnn_model import P_Net, cls_ohem, bbox_ohem, landmark_ohem, cal_accuracy
from train_models.utils import image_color_distort, random_flip_images
import tensorflow as tf
import os, sys, time
import random
from train_models.utils import get_dataset, load_and_get_normalization_img
tf.enable_eager_execution()


def cls_loss(pred, label):
    cls_prob = tf.squeeze(pred, [1, 2])
    return cls_ohem(cls_prob,label)

def bbox_loss(pred, bbox_target, label):
    bbox_pred = tf.squeeze(pred, [1, 2])
    return bbox_ohem(bbox_pred,bbox_target,label)

def landmark_loss(pred, landmark_target, label):
    landmark_pred = tf.squeeze(pred, [1, 2])
    return landmark_ohem(landmark_pred, landmark_target, label)

def cls_acc(cls_pred, labels):
    cls_pred = tf.squeeze(cls_pred, [1, 2])
    return cal_accuracy(cls_pred, labels)


def total_loss(model, images, labels, bboxes, landmarks):
    """
    Return
    --------------------
        (total_loss, cls_loss, bbox_loss, landmark_loss)
    """
    pred = model(images)
    c_loss = cls_loss(pred[0], labels)
    b_loss = bbox_loss(pred[1], bboxes, labels)
    l_loss = landmark_loss(pred[2], landmarks, labels)
    # losses = tf.keras.losses.get_regularization_losses()
    # l2_loss = tf.add_n(losses)
    return c_loss + 0.5 * b_loss + 0.5 * l_loss #+ l2_loss

def grad(model, images, labels, bboxes, landmarks):
    with tf.GradientTape() as tape:
        # must execute model(x) in the context of tf.GradientTape()
        loss_value = total_loss(model, images, labels, bboxes, landmarks)
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
    batch_size = 512
    total_num, train_dataset = get_dataset("../data/imglists/PNet", batch_size=batch_size)

    optimizer = tf.train.AdamOptimizer()
    # callbacks = [tf.keras.callbacks.ModelCheckpoint("../data/ultramodern_model/PNet/pnet.h5",
    #                                                 monitor="multi_loss", 
    #                                                 save_best_only=True),                                          
    #             ]
    # # 计算损失时会用到额外数据，所以只能自己写training loop
    # losses = {
    #     "cls_output": "categorical_crossentropy",
    #     "bbox_output": "categorical_crossentropy",
    # }
    # lossWeights = {"cls_output": 1.0, "bbox_output": 1.0}

    # model.compile(optimizer, loss=multi_loss, metrics=[cls_acc])
    # model.fit(train_dataset, epochs=30, steps_per_epoch=total_num//batch_size, callbacks=callbacks)


    os.makedirs(prefix, exist_ok=True)
    checkpoint_prefix = os.path.join(prefix, "ckpt")
    root = tf.train.Checkpoint(optimizer=optimizer, model=model, optimizer_step=tf.train.get_or_create_global_step()) 

    display_step = 100

    #estimate time left
    now = time.time()
    pre = now

    print("start training")
    for epoch in range(end_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.Mean()

        for i, train_batch in enumerate(train_dataset):
            images, target_batch = train_batch
            labels, bboxes, landmarks = target_batch
            images = image_color_distort(images)
            # images, landmarks = random_flip_images(images, labels, landmarks)
            total_loss, grads = grad(model, images, labels, bboxes, landmarks)
            optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step=tf.train.get_or_create_global_step())

            display_pred = model(images)
            acc_value = cls_acc(display_pred[0], labels)
            c_loss = cls_loss(display_pred[0], labels)
            b_loss = bbox_loss(display_pred[1], bboxes, labels)
            l_loss = landmark_loss(display_pred[2], landmarks, labels)
            # l2_loss = tf.add_n(tf.losses.get_regularization_losses)
            # total_loss, c_loss, b_loss, l_loss = loss_value
            epoch_loss_avg(total_loss)
            epoch_accuracy_avg(acc_value)

            if i % display_step == 0:
                now = time.time()
                total_steps = total_num // batch_size
                remaining_time = (now - pre) * (total_steps - i) / display_step // 60
                sys.stdout.write("\r>> {0} of {1} steps done. Estimated remaining time: {2} mins. \
loss_value: {3:.3f} acc: {4:.3f}. cls_loss: {5:.3f}, bbox_loss: {6:.3f}, landmark_loss: {7:.3f}".format(i, 
                                                                                                        total_steps, 
                                                                                                        remaining_time,
                                                                                                        total_loss.numpy(),
                                                                                                        acc_value.numpy(),
                                                                                                        c_loss.numpy(),
                                                                                                        b_loss.numpy(),
                                                                                                        l_loss.numpy()))
                sys.stdout.flush()  
                pre = now

        print("\nEpoch {0}: Loss: {1} Accuracy: {2}".format(epoch, epoch_loss_avg.result(), epoch_accuracy_avg.result()))
        print("VALIDATION: try to predict a pos pic for cls_prob: ", model(tf.expand_dims(load_and_get_normalization_img("test/not test/7.jpg"), axis=0))[0])
        print("VALIDATION: try to predict a neg pic for cls_prob: ", model(tf.expand_dims(load_and_get_normalization_img("test/not test/778.jpg"), axis=0))[0])

        # save model
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
