from keras.models import Model
from keras import optimizers
from os.path import join, isdir
from os import makedirs, listdir
import cv2

from model import two_streams
from methods import rgb2pca
from dataset import *

import numpy as np


def two_streams_rgb():
    dataset_train = '/home/cic/datasets/ImageNet/train/'
    dataset_test = '/home/cic/datasets/ImageNet/validation/'
    save_dir = '/home/nsallent/output/saved_models/'
    model_name = 'two_streams_rgb'

    # for folder in listdir(dataset_train):
    #     y_train_all.extend([folder] * len(listdir(dataset_train + folder)))
    #     x_train_all.extend([cv2.imread(dataset_train + folder + '/' + im) for im in listdir(dataset_train + folder)])

    x_train = []
    y_train = []

    for folder in listdir(dataset_train):
        if folder in classes_values:
            for im in listdir(dataset_train + folder):
                im_pca = rgb2pca(cv2.imread(dataset_train + folder + '/' + im))
                x_train.append(im_pca)
                y_train.append(folder)

    print(np.unique(y_train))

    x_test = []
    y_test = []

    for folder in listdir(dataset_test):
        if folder in classes_values:
            for im in listdir(dataset_test + folder):
                im_pca = rgb2pca(cv2.imread(dataset_test + folder + '/' + im))
                x_test.append(im_pca)
                y_test.append(folder)

    print(np.unique(y_test))

    # y_test_all.extend([folder] * len(listdir(dataset_test + folder)))
    # x_test_all.extend([cv2.imread(dataset_test + folder + '/' + im) for im in listdir(dataset_test + folder)])
    #
    # x_test = []
    # y_test = []
    #
    # for value in classes_values:
    #     for im, label in zip(x_test_all, y_test_all):
    #         if label == value:
    #             im_pca = rgb2pca(im)
    #             x_test.append(im_pca)
    #             y_test.append(label)

    print('All images loaded.')

    x, im_input, INP_SHAPE, DIM_ORDERING = two_streams()

    model = Model(input=im_input,
                  output=[x])

    opt = optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy')

    model.fit_generator(
        (x_train, y_train),
        steps_per_epoch=2000,
        epochs=50,
        validation_data=(x_test, y_test))

    # Save model and weights
    if not isdir(save_dir):
        makedirs(save_dir)
    model_path = join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
