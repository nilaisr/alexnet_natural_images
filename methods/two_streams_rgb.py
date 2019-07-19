from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers
from os.path import join, isdir
from os import makedirs, listdir

from model import two_streams
from methods import rgb2pca
from dataset import *


def two_streams_rgb():
    dataset_train = '/home/cic/datasets/ImageNet/train/'
    dataset_test = '/home/cic/datasets/ImageNet/validation/'
    save_dir = '/home/nsallent/output/saved_models/'
    model_name = 'two_streams_rgb'

    input_size = 224

    classes_train = []

    for folder in listdir(dataset_train):
        if folder in classes_values:
            classes_train.append(folder)

    classes_test = []

    for folder in listdir(dataset_test):
        if folder in classes_values:
            classes_test.append(folder)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        dataset_train,
        target_size=(input_size, input_size),
        classes=classes_train[:3],
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        dataset_test,
        target_size=(input_size, input_size),
        classes=classes_test[:3],
        class_mode='categorical')

    output, im_input, input_shape = two_streams()

    ## TO-DO: PUT PREPROCESS OF IMAGES!!!!!

    model = Model(inputs=im_input,
                  outputs=output)

    print(model.summary())

    opt = optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy')

    model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=80)

    # Save model and weights
    if not isdir(save_dir):
        makedirs(save_dir)
    model_path = join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate_generator(validation_generator)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
