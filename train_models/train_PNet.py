#coding:utf-8
from train_models.mtcnn_model import P_Net
from train_models.train import train


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
    net_factory = P_Net
    train(net_factory,prefix, end_epoch, base_dir, display=display, base_lr=lr) 

if __name__ == '__main__':
    #data path
    base_dir = '../DATA/imglists/PNet'
    model_name = 'MTCNN_no_distort'
    #with landmark
    model_path = '../data/%s_model/PNet_landmark/PNet' % model_name
            
    prefix = model_path
    end_epoch = 30
    display = 100
    lr = 0.001
    train_PNet(base_dir, prefix, end_epoch, display, lr)
