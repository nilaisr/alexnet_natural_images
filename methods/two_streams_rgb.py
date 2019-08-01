from keras.preprocessing.image import ImageDataGenerator
from keras import Model, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from os.path import join, isdir
from os import makedirs, listdir
import numpy as np
import math

from model import two_streams
from methods import rgb2pca
from dataset import *


def two_streams_rgb():
    dataset_train = '/home/cic/datasets/ImageNet/train/'
    dataset_test = '/home/cic/datasets/ImageNet/validation/'
    save_dir = '/home/nsallent/output/saved_models/'
    model_name = 'two_streams_rgb'

    input_size = 224
    batch_size = 512

    classes_train = []

    for folder in listdir(dataset_train):
        if folder in classes_values:
            classes_train.append(folder)

    classes_test = []

    for folder in listdir(dataset_test):
        if folder in classes_values:
            classes_test.append(folder)

    def color_transformation(image):
        image = np.array(image)
        pca_image = rgb2pca(image)
        scalers = {}
        for i in range(pca_image.shape[0]):
            scalers[i] = MinMaxScaler((0, 255))
            pca_image[i, :, :] = scalers[i].fit_transform(pca_image[i, :, :])
        pca_image = np.around(pca_image)
        return pca_image

    train_datagen = ImageDataGenerator(preprocessing_function=color_transformation)

    test_datagen = ImageDataGenerator(preprocessing_function=color_transformation)

    train_generator = train_datagen.flow_from_directory(dataset_train,
                                                        target_size=(input_size, input_size),
                                                        batch_size=batch_size,
                                                        classes=classes_train[:20])

    validation_generator = test_datagen.flow_from_directory(dataset_test,
                                                            target_size=(input_size, input_size),
                                                            batch_size=batch_size,
                                                            classes=classes_test[:20])

    output, im_input, input_shape = two_streams()

    model = Model(inputs=im_input,
                  outputs=output)

    print(model.summary())

    num_train_samples = train_generator.samples
    # num_test_samples = validation_generator.samples
    train_steps_per_epoch = min(math.ceil(num_train_samples/(batch_size)), 100)/5
    # test_steps_per_epoch = math.ceil(num_test_samples/(batch_size*25))

    sgd = optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    rmsp = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
    reduceLRPlateau = ReduceLROnPlateau(monitor='categorical_crossentropy', factor=0.1, patience=10, verbose=0,
                                        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    earlystopping = EarlyStopping(monitor='categorical_crossentropy', min_delta=0, patience=0, verbose=0, mode='auto',
                                  baseline=None, restore_best_weights=False)

    model.fit_generator(train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=55,
                        shuffle='batch',
                        validation_data=validation_generator,
                        validation_steps=70,
                        callbacks=[checkpointer, reduceLRPlateau, earlystopping],
                        use_multiprocessing=True)

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
