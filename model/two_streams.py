from keras.layers import ZeroPadding2D, Conv2D, Dense, Dropout, Flatten, Activation, MaxPooling2D, Input
from keras.layers import BatchNormalization, Concatenate, Multiply
import numpy as np

np.random.seed(1000)


def conv2D_bn(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    """Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    bn_axis = 3
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu', name=name)(x)
    return x


def two_streams():
    num_classes = 444  # number of classes
    dropout = 0.5

    data_format = 'channels_last'
    input_shape = (224, 224, 3)  # 3 - Number of RGB Colours

    img_input = Input(shape=input_shape)

    # Channel 1 - Conv Net Layer 1
    x = conv2D_bn(img_input, 48, 11, 11, strides=(4, 4), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(x)

    # Channel 2 - Conv Net Layer 1
    y = conv2D_bn(img_input, 48, 11, 11, strides=(4, 4), padding='same')
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(y)

    # Channel 1 - Conv Net Layer 2
    x = conv2D_bn(x, 128, 5, 5, strides=(1, 1), padding='valid')
    x = ZeroPadding2D(padding=(2, 2), data_format=data_format)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(x)

    # Channel 2 - Conv Net Layer 2
    y = conv2D_bn(y, 128, 5, 5, strides=(1, 1), padding='valid')
    y = ZeroPadding2D(padding=(2, 2), data_format=data_format)(y)
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(y)

    # Channel 1 - Conv Net Layer 3
    x = Conv2D(192, (3, 3), strides=(1, 1), padding='same')(x)

    # Channel 2 - Conv Net Layer 3
    y = Conv2D(192, (3, 3), strides=(1, 1), padding='same')(y)

    # Channel 1 - Conv Net Layer 4
    x1 = Concatenate()([x, y])
    x1 = Conv2D(192, (3, 3), strides=(1, 1), padding='same')(x1)

    # Channel 2 - Conv Net Layer 4
    y1 = Concatenate()([x, y])
    y1 = Conv2D(192, (3, 3), strides=(1, 1), padding='same')(y1)

    # Channel 1 - Conv Net Layer 5
    x2 = Concatenate()([x1, y1])
    x2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x2)
    x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(x2)

    # Channel 2 - Conv Net Layer 5
    y2 = Concatenate()([x1, y1])
    y2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(y2)
    y2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(y2)

    # # Channel 1 - Cov Net Layer 6
    x3 = Multiply()([x2, y2])
    x3 = Flatten()(x3)
    x3 = Dense(2048, activation='relu')(x3)
    x3 = Dropout(dropout)(x3)

    # Channel 2 - Cov Net Layer 6
    y3 = Multiply()([x2, y2])
    y3 = Flatten()(y3)
    y3 = Dense(2048, activation='relu')(y3)
    y3 = Dropout(dropout)(y3)

    # Channel 1 - Cov Net Layer 7
    x4 = Multiply()([x3, y3])
    x4 = Dense(2048, activation='relu')(x4)
    x4 = Dropout(dropout)(x4)

    # Channel 2 - Cov Net Layer 7
    y4 = Multiply()([x3, y3])
    y4 = Dense(2048, activation='relu')(y4)
    y4 = Dropout(dropout)(y4)

    # Final Channel - Cov Net 9
    xy = Multiply()([x4, y4])
    output = Dense(num_classes, activation='softmax')(xy)

    return output, img_input, input_shape
