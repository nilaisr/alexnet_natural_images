from keras.models import Model
from keras.utils.visualize_util import plot
from model import create_model


def check_print():
    # Create the Model
    xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

    # Create a Keras Model - Functional API
    model = Model(input=img_input,
                  output=[xy])
    model.summary()

    # Save a PNG of the Model Build
    plot(model, to_file='./Model/AlexNet_Original.png')

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')
    print('Model Compiled')
