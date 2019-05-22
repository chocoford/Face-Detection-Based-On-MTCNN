import os, random
import tensorflow as tf

def get_dataset(path, batch_size=256, ratios=[1, 3, 1, 1]):
    """
    Parameter
    --------------------
        path: path of data source.
        batch_size
        ratios: [pos_ratio, neg_ratio, part_ratio, landmark_ratio]

    Return
    --------------------
        tuple: (num of samples, dataset)
    """
    net = path[-4:]
    item = 'train_%s_landmark.txt' % net
    dataset_dir = os.path.join(path, item)

    imagelist = open(dataset_dir, 'r')

    all_image_paths = []
    all_image_labels = []
    all_image_bboxes = []
    all_image_landmarks = []

    pos_lines = []
    neg_lines = []
    part_lines = []
    landmark_lines = []

    lines = imagelist.readlines()

    for line in lines:
        info = line.strip().split(' ')
        if int(info[1]) == 1:
            pos_lines.append(line)
        elif int(info[1]) == 0:
            neg_lines.append(line)
        elif int(info[1]) == -1:
            part_lines.append(line)
        elif int(info[1]) == -2:
            landmark_lines.append(line)
        else:
            print("unknown label.")
            exit(-1)
    print("There are {} pos samples, {} neg samples, {} part samples and {} landmark samples \
        before keeping ratios. {} samples in total".format(len(pos_lines),
                                                           len(neg_lines), 
                                                           len(part_lines), 
                                                           len(landmark_lines),
                                                           len(lines)))
    base_num = min(len(pos_lines), len(neg_lines), len(part_lines), len(landmark_lines))
    pos_num = ratios[0] * base_num
    neg_num = ratios[1] * base_num
    part_num = ratios[2] * base_num
    landmark_num = ratios[3] * base_num    

    random.shuffle(pos_lines)
    random.shuffle(neg_lines)
    random.shuffle(part_lines)
    random.shuffle(landmark_lines)

    shuffled_lines = pos_lines[:pos_num] + neg_lines[:neg_num] + part_lines[:part_num] + landmark_lines[:landmark_num]
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

    print("There are {} pos samples, {} neg samples, {} part samples and {} landmark samples \
        after keeping ratios. {} samples in total".format(all_image_labels.count(1), 
                                                          all_image_labels.count(0), 
                                                          all_image_labels.count(-1), 
                                                          all_image_labels.count(-2),
                                                          len(shuffled_lines)))

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
    target_ds = tf.data.Dataset.zip((label_ds, bbox_ds, landmark_ds))

    image_label_bbox_landmarks_ds = tf.data.Dataset.zip((image_ds, target_ds))
    # print(image_label_bbox_landmarks_ds)

    ds = image_label_bbox_landmarks_ds.cache()
    ds = ds.shuffle(buffer_size=10000)
    # ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print("get dataset done.")

    return len(all_image_paths), ds


if __name__ == "__main__":
    get_dataset("../data/imglists/RNet")