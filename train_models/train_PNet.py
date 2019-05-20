#coding:utf-8
from train_models.mtcnn_model import P_Net, cls_ohem, bbox_ohem, landmark_ohem, cal_accuracy
from train_models.train import train
import tensorflow as tf
import os, sys, time
import random
from prepare_data.read_tfrecord_v2 import read_multi_tfrecords,read_single_tfrecord
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

def loss(model, images, labels, bboxes, landmarks):
    c_loss = cls_loss(model, images, labels)
    b_loss = bbox_loss(model, images, bboxes, labels)
    l_loss = landmark_loss(model, images, landmarks, labels)
    return c_loss + 0.5 * b_loss + 0.5 * l_loss


def grad(model, images, labels, bboxes, landmarks):
    with tf.GradientTape() as tape:
        loss_value = loss(model, images, labels, bboxes, landmarks)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def get_dataset(path, batch_size=256):
    """

    Return
    --------------------
        tuple: (num of samples, dataset)
    """
    net = "PNet"
    item = 'train_%s_landmark.txt' % net
    dataset_dir = os.path.join(path, item)

    imagelist = open(dataset_dir, 'r')

    all_image_paths = []
    all_image_labels = []
    all_image_bboxes = []
    all_image_landmarks = []

    shuffled_lines = imagelist.readlines() 
    random.shuffle(shuffled_lines)
    for line in shuffled_lines:
        info = line.strip().split(' ')
        all_image_paths.append(info[0])
        all_image_labels.append(float(info[1]))
        if len(info) == 6:
            all_image_bboxes.append((float(info[2]), float(info[3]), float(info[4]), float(info[5])))
            all_image_landmarks.append((0., 0., 0., 0., 0., 0., 0., 0., 0., 0.))
        elif len(info) == 12:
            all_image_bboxes.append((0., 0., 0., 0.))
            all_image_landmarks.append((float(info[2]), float(info[3]), float(info[4]), float(info[5]), 
                                        float(info[6]), float(info[7]), float(info[8]), float(info[9]),
                                        float(info[10]), float(info[11])
                                        ))
        else:
            all_image_bboxes.append((0., 0., 0., 0.))
            all_image_landmarks.append((0., 0., 0., 0., 0., 0., 0., 0., 0., 0.))

    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [12, 12])
        image /= 255.0  # normalize to [0,1] range
        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)
        # return tf.zeros((12, 12, 3))

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    # print(path_ds)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(all_image_labels)
    bbox_ds = tf.data.Dataset.from_tensor_slices(all_image_bboxes)
    landmark_ds = tf.data.Dataset.from_tensor_slices(all_image_landmarks)

    image_label_bbox_landmarks_ds = tf.data.Dataset.zip((image_ds, label_ds, bbox_ds, landmark_ds))
    # print(image_label_bbox_landmarks_ds)

    ds = image_label_bbox_landmarks_ds.cache()
    ds = ds.shuffle(buffer_size=10000)
    # ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print("get dataset done.")

    return len(all_image_paths), ds


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

    display_step = 100

    optimizer = tf.train.MomentumOptimizer(lr, 0.9)

    #estimate time left
    now = time.time()
    pred = now

    print("start training")
    for epoch in range(end_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        # epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for i, (input_images, labels, bboxes, landmarks) in enumerate(train_dataset):
            
            loss_value, grads = grad(model, input_images, labels, bboxes, landmarks)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # cls_pred = tf.squeeze(model(input_images)[0], [1, 2])
            epoch_loss_avg(loss_value)
            # epoch_accuracy(labels, cls_pred[:, 1])

            if i % display_step == 0:
                now = time.time()
                total_steps = total_num // batch_size
                remaining_time = (now - pred) * (total_steps - i) / display_step // 60
                sys.stdout.write("\r>> {0} of {1} steps done. Estimated remaining time: {3} mins. loss_value: {2}\n".format(i, 
                                                                                                                    total_steps, 
                                                                                                                    loss_value.numpy(),
                                                                                                                    remaining_time))
                sys.stdout.flush()  
                pred = now

        # if epoch % 50 == 0:
        print("Epoch {0}: Loss: {1}".format(epoch, epoch_loss_avg.result()))

        # save model
        checkpoint_dir = "../data/ultramodern_model/PNet"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        root = tf.train.Checkpoint(optimizer=optimizer,
                                model=model)
        root.save(checkpoint_prefix)

if __name__ == '__main__':
    #data path
    base_dir = '../DATA/imglists/PNet'
    model_name = 'MTCNN'
    #with landmark
    model_path = '../data/%s_model/PNet_landmark/PNet' % model_name
            
    prefix = model_path
    end_epoch = 30
    display = 100
    lr = 0.001
    train_PNet(base_dir, prefix, end_epoch, display, lr)
