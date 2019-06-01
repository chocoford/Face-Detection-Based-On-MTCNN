import os
from PIL import Image
list_path = "/Users/zhangdefu/Developer/Tensorflow/Face Detection Based On MTCNN/Face-Detection-Based-On-MTCNN/data/FDDB_OUTPUT/Fold_all.txt"
jpg_dir = "/Users/zhangdefu/Developer/Tensorflow/Face Detection Based On MTCNN/evaluation/FDDB"
ppm_path = "/Users/zhangdefu/Developer/Tensorflow/Face Detection Based On MTCNN/evaluation/FDDB-ppm"

f = open(list_path, 'r')
# os.mkdir(ppm_path)
for im_path in f.readlines():
    im_path = im_path[:-1]
    dir_path = im_path[:15]
    os.makedirs(os.path.join(ppm_path, dir_path), exist_ok=True)
    origin_path = os.path.join(jpg_dir, im_path+".jpg")
    save_path = os.path.join(ppm_path, im_path+".ppm")
    img = Image.open(origin_path)
    img.save(save_path)

