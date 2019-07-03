from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers
from os.path import join, isdir
from os import makedirs

from model import two_streams
from dataset import *


def two_streams_rgb():
    dataset = '/home/cic/datasets/ImageNet/'
    save_dir = '/home/nsallent/output/saved_models/'
    model_name = 'two_streams_rgb'

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    (x_train_all, y_train_all) = train_datagen.flow_from_directory(
        dataset + 'train/',
        target_size=(227, 227))

    x_train = []
    y_train = []

    for value in classes_values:
        for im, label in zip(x_train_all, y_train_all):
            if label == value:
                x_train.append(im)
                y_train.append(label)
        #x_train, y_train = [im, label for (im, label) in zip(x_train,y_train) if label == value]

    (x_test_all, y_test_all) = test_datagen.flow_from_directory(
        dataset + 'test/',
        target_size=(227, 227))

    x_test = []
    y_test = []

    for value in classes_values:
        for im, label in zip(x_test_all, y_test_all):
            if label == value:
                x_test.append(im)
                y_test.append(label)


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
