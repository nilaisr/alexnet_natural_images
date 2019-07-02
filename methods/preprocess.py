from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers
from os.path import join, isdir
from os import makedirs

from model import two_streams


def preprocess():
    dataset = 'home/cic/datasets/ImageNet/'
    save_dir = 'home/nsallent/output/saved_models/'

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_datagen.flow_from_directory(dataset + 'train/')
    test_datagen.flow_from_directory(dataset + 'test/')

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    x, im_input, INP_SHAPE, DIM_ORDERING = two_streams()

    model = Model(input=im_input,
                  output=[x])

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)

    sgd = optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy')


    # Save model and weights
    if not isdir(save_dir):
        makedirs(save_dir)
    model_path = join(save_dir, )
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])