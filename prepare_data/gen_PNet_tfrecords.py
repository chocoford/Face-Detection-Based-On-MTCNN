"""
将与处理好的图片和txt通通转换成tfrecord文件，以便提高训练时的速度。
"""

#coding:utf-8
import os
import random
import sys
import time

import tensorflow as tf

from prepare_data.tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """
    Loads data from image and annotations files and add them to a TFRecord.

    Parameters
    --------------------------
        filename: Dataset directory;
        name: Image name to add to the TFRecord;
        tfrecord_writer: The TFRecord writer to use for writing.
    """
    #image_example dict contains image's info
    image_data, _, _ = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, net):
    return '%s/train_PNet_landmark.tfrecord' % (output_dir)
    

def run(dataset_dir, net, output_dir, name='MTCNN', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    
    #tfrecord name 
    tf_filename = _get_output_filename(output_dir, name, net)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
    dataset = get_dataset(dataset_dir, net=net)
    # filenames = dataset['filename']
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        #random.seed(64)
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    print('start writing the data to tfrecord.')
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i+1) % 100 == 0:
                sys.stdout.write('\r>> %d/%d images has been converted' % (i+1, len(dataset)))
            sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')


def get_dataset(dir, net='PNet'):
    """
        获得数据
    Parameter
    ---------------
        dir: 数据所在目录
        net: 网络类别
    Return
    --------------
        dataset: list<dict>
    """
    item = 'imglists/PNet/train_%s_landmark.txt' % net
    
    dataset_dir = os.path.join(dir, item)
    #print(dataset_dir)
    imagelist = open(dataset_dir, 'r')

    dataset = []
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        #print(data_example['filename'])
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0        
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) == 12:
            bbox['xlefteye'] = float(info[2])
            bbox['ylefteye'] = float(info[3])
            bbox['xrighteye'] = float(info[4])
            bbox['yrighteye'] = float(info[5])
            bbox['xnose'] = float(info[6])
            bbox['ynose'] = float(info[7])
            bbox['xleftmouth'] = float(info[8])
            bbox['yleftmouth'] = float(info[9])
            bbox['xrightmouth'] = float(info[10])
            bbox['yrightmouth'] = float(info[11])
            
        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset


if __name__ == '__main__':
    dir = '../DATA/'
    net = 'PNet'
    output_directory = '../DATA/imglists/PNet'
    run(dir, net, output_directory, shuffling=True)
