import cv2
from os import listdir
from os.path import isfile, join


def two_streams_rgb():
    dataset = 'home/cic/datasets/ImageNet/'
    all_images = []

    for folder in listdir(dataset + 'train/'):
        images = [f for f in listdir(folder) if isfile(join(folder, f))]
        im_class = images[0].split('_')
        for image in images:
            all_images.append(dataset + 'train/' + im_class + image)

    for image in all_images:
    x_train = cv2.imread(image)
