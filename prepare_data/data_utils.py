

def IoU(box, bboxes):
    """
    Caculate IoU between detect and ground truth boxes
    :param crop_box:numpy array (4, )
    :param bboxes:numpy array (n, 4):x1, y1, x2, y2
    :return:
    numpy array, shape (n, ) Iou
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
    xx1 = np.maximum(box[0], bboxes[:, 0])
    yy1 = np.maximum(box[1], bboxes[:, 1])
    xx2 = np.minimum(box[2], bboxes[:, 2])
    yy2 = np.minimum(box[3], bboxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    over = inter / (box_area + areas - inter)

    return over




if __name__ == '__main__':
    dir = '/media/thinkjoy/新加卷/dataset/widerface/wider_face_split/wider_face_train_bbx_gt.txt'
    base_dir = '/media/thinkjoy/新加卷/dataset/widerface'
    data = read_annotation(base_dir, dir)
    print('\n')
    print(data['images'])
    print("============")
    print('\n')
    print(data['bboxes'])





