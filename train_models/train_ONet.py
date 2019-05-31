#coding:utf-8
from train_models.mtcnn_model import O_Net, cls_ohem, bbox_ohem, landmark_ohem, cal_accuracy
from train_models.utils import get_dataset, load_and_get_normalization_img
import tensorflow as tf
import time, os, sys
tf.enable_eager_execution()

def cls_loss(pred, label):
    return cls_ohem(pred, label)

def bbox_loss(pred, bbox_target, label):
    return bbox_ohem(pred, bbox_target, label)

def landmark_loss(pred, landmark_target, label):
    return landmark_ohem(pred, landmark_target, label)


def cls_acc(cls_pred, labels):
    return cal_accuracy(cls_pred, labels)


def get_total_loss(pred, labels, bboxes, landmarks):
    c_loss = cls_loss(pred[0], labels)
    b_loss = bbox_loss(pred[1], bboxes, labels)
    l_loss = landmark_loss(pred[2], landmarks, labels)
    return c_loss * 0.5 + 0.5 * b_loss + l_loss

def grad(model, images, labels, bboxes, landmarks):
    with tf.GradientTape() as tape:
        pred = model(images, True)
        loss_value = get_total_loss(pred, labels, bboxes, landmarks)
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
    total_num, train_dataset, validation_dataset = get_dataset(base_dir, batch_size=batch_size)
    optimizer = tf.train.AdamOptimizer()


    # prepare for save
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    root = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # if continue training process
    # root.restore(tf.train.latest_checkpoint(prefix)).assert_existing_objects_matched()


    now = time.time()
    pre = now

    #logs
    fpath = prefix+"/logs.txt"

    print("start training")
    for epoch in range(end_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy_avg = tf.keras.metrics.Mean()
        epoch_val_loss_avg = tf.keras.metrics.Mean()
        epoch_val_acc_avg = tf.keras.metrics.Mean()

        for i, train_batch in enumerate(train_dataset):
            images, target_batch = train_batch
            labels, bboxes, landmarks = target_batch
            loss_value, grads = grad(model, images, labels, bboxes, landmarks)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            display_pred = model(images)
            acc_value = cls_acc(display_pred[0], labels)
            c_loss = cls_loss(display_pred[0], labels)
            b_loss = bbox_loss(display_pred[1], bboxes, labels)
            l_loss = landmark_loss(display_pred[2], landmarks, labels)
            epoch_loss_avg(loss_value)
            epoch_accuracy_avg(acc_value)

            if i % display_step == 0:
                now = time.time()
                total_steps = total_num // batch_size
                remaining_time = (now - pre) * (total_steps - i) / display_step // 60
                sys.stdout.write("\r>> {0} of {1} steps done. Estimated remaining time: {2} mins. \
loss_value: {3:.3f} acc: {4:.3f}. cls_loss: {5:.3f}, bbox_loss: {6:.3f}, landmark_loss: {7:.3f}".format(i, 
                                                                                                        total_steps, 
                                                                                                        remaining_time,
                                                                                                        loss_value.numpy(),
                                                                                                        acc_value.numpy(),
                                                                                                        c_loss.numpy(),
                                                                                                        b_loss.numpy(),
                                                                                                        l_loss.numpy()))
                sys.stdout.flush()  
                pre = now

        for i, val_batch in enumerate(validation_dataset):
            images, target_batch = val_batch
            labels, bboxes, landmarks = target_batch
            display_pred = model(images)
            acc_value = cls_acc(display_pred[0], labels)
            val_loss = get_total_loss(display_pred, labels, bboxes, landmarks)
            epoch_val_loss_avg(val_loss)
            epoch_val_acc_avg(acc_value)

        print("\nEpoch {0}: Loss: {1} Accuracy: {2} val_loss: {3} val_acc: {4}".format(epoch, 
                                                                                       epoch_loss_avg.result(), 
                                                                                       epoch_accuracy_avg.result(),
                                                                                       epoch_val_loss_avg.result(),
                                                                                       epoch_val_acc_avg.result()))
        print("VALIDATION: try to predict a pos pic for cls_prob: ", model(tf.expand_dims(load_and_get_normalization_img("test/not test/7.jpg", 48), axis=0))[0])
        print("VALIDATION: try to predict a neg pic for cls_prob: ", model(tf.expand_dims(load_and_get_normalization_img("test/not test/10001.jpg", 48), axis=0))[0])

        # save model
        save_path = root.save(checkpoint_prefix)
        print("save prefix is {}".format(save_path))

        #save logs
        with open(fpath, 'a+') as f:
            f.write("{0}: Loss: {1} Accuracy: {2} Val_Loss: {3} Val_Acc: {4}\n".format(save_path, 
                                                                                  epoch_loss_avg.result(),
                                                                                  epoch_accuracy_avg.result(),
                                                                                  epoch_val_loss_avg.result(),
                                                                                  epoch_val_acc_avg.result()))



if __name__ == '__main__':

    base_dir = '../DATA/imglists/ONet'
    model_name = 'ultramodern'
    model_path = '../data/%s_model/ONet' % model_name
    prefix = model_path
    end_epoch = 40
    display = 10
    lr = 0.001
    train_ONet(base_dir, prefix, end_epoch, display, lr)
