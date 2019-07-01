from keras.models import Model
from keras.utils.vis_utils import plot_model
from model import two_streams


def check_print():
    # Create the Model
    xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = two_streams()

    # Create a Keras Model - Functional API
    model = Model(input=img_input,
                  output=[xy])
    model.summary()

    # Save a PNG of the Model Build
    plot_model(model, to_file='./Model/AlexNet_Original.png')

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')
    print('Model Compiled')
