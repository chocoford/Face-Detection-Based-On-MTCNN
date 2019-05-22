import numpy as np
import os
import cv2

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        predicted boxes
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):
    """Convert bbox to square

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox



def load_wider_face_gt_boxes(fpath, basedir=""): 
    """
    get the information about all groud true of images in wider face datasets.
    Parameter
    ---------------------
        fpath: the path of the txt file.
    Return
    ---------------------
        dict:(path:String, gt_boxes:Numpy) gt_boxes: [x_min, y_min, x_max, y_max]s of all images.
    """
    f = open(fpath)
    data = f.read()
    f.close()

    # header = ["x1", "y1", "w", "h", "blur", "expression", "illumination", "invalid", "occlusion", "pose"]
    lines = data.split('\n')
    gt_data = {}
    i = 0
    while True:
        gt_box_num = 1 if int(lines[i+1]) == 0 else int(lines[i+1])
        gt_pos = np.zeros((gt_box_num, 4))
        for j, bbox_list in enumerate([x.split(' ')[:4] for x in lines[i+2:i+gt_box_num+2]]):
            gt_pos[j] = [float(x) for x in bbox_list]
            gt_pos[j, 2] += gt_pos[j, 0] - 1
            gt_pos[j, 3] += gt_pos[j, 1] - 1
        gt_data[os.path.join(basedir, lines[i])] = gt_pos
        i += gt_box_num + 2
        if i >= len(lines) - 1: #最后一行有一个换行
            break
    
    return gt_data




def read_annotation(base_dir, label_path):
    """
    read label file
    :param dir: path
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    gt_data = load_wider_face_gt_boxes(label_path)
    for path, bboxes in gt_data.items():
        imagepath = base_dir + '/WIDER_train/images/' + path
        images.append(imagepath)


    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        # im = cv2.imread(imagepath)
        # h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums) if int(nums) > 0 else 1):
            # text = ''
            # text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            # text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            # text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            # f.write(text + '\n')
        bboxes.append(one_image_bboxes)


    data['images'] = images#all images
    data['bboxes'] = bboxes#all image bboxes
    # f.close()
    return data

def read_and_write_annotation(base_dir, dir):
    """
    read label file
    :param dir: path
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    labelfile = open(dir, 'r')
    f = open('/home/thinkjoy/data/mtcnn_data/imagelists/train.txt', 'w')
    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        im = cv2.imread(imagepath)
        h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums)):
            text = ''
            text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2] - 1
            ymax = ymin + face_box[3] - 1
            text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            f.write(text + '\n')
        bboxes.append(one_image_bboxes)


    data['images'] = images
    data['bboxes'] = bboxes
    f.close()
    return data

def get_path(base_dir, filename):
    return os.path.join(base_dir, filename)


