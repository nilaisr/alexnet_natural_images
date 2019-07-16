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
    bn_axis = 1
    # bn_axis = 3
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu', name=name)(x)
    return x


def two_streams():
    # global constants
    num_classes = 444  # number of classes
    dropout = 0.5

    data_format = 'channels_last'
    concat_axis = 3
    input_shape = (227, 227, 3)  # 3 - Number of RGB Colours

    img_input = Input(shape=input_shape)

    # Channel 1 - Conv Net Layer 1
    x = conv2D_bn(img_input, 3, 11, 11, strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), data_format=data_format)(x)
    x = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x)

    # Channel 2 - Conv Net Layer 1
    y = conv2D_bn(img_input, 3, 11, 11, strides=(1, 1), padding='same')
    y = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), data_format=data_format)(y)
    y = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y)

    # Channel 1 - Conv Net Layer 2
    x = conv2D_bn(x, 48, 55, 55, strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(x)
    x = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x)

    # Channel 2 - Conv Net Layer 2
    y = conv2D_bn(y, 48, 55, 55, strides=(1, 1), padding='same')
    y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(y)
    y = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y)

    # Channel 1 - Conv Net Layer 3
    x = conv2D_bn(x, 128, 27, 27, strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(x)
    x = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x)

    # Channel 2 - Conv Net Layer 3
    y = conv2D_bn(y, 128, 27, 27, strides=(1, 1), padding='same')
    y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(y)
    y = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y)

    # Channel 1 - Conv Net Layer 4
    x1 = Concatenate()([x, y])
    x1 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x1)
    x1 = conv2D_bn(x1, 192, 13, 13, strides=(1, 1), padding='same')

    # Channel 2 - Conv Net Layer 4
    y1 = Concatenate()([x, y])
    y1 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y1)
    y1 = conv2D_bn(y1, 192, 13, 13, strides=(1, 1), padding='same')

    # Channel 1 - Conv Net Layer 5
    x2 = Concatenate()([x1, y1])
    x2 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x2)
    x2 = conv2D_bn(x2, 192, 13, 13, strides=(1, 1), padding='same')

    # Channel 2 - Conv Net Layer 5
    y2 = Concatenate()([x1, y1])
    y2 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y2)
    y2 = conv2D_bn(y2, 192, 13, 13, strides=(1, 1), padding='same')

    # Channel 1 - Cov Net Layer 6
    x3 = conv2D_bn(x2, 128, 27, 27, strides=(1, 1), padding='same')
    x3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(x3)
    x3 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x3)

    # Channel 2 - Cov Net Layer 6
    y3 = conv2D_bn(y2, 128, 27, 27, strides=(1, 1), padding='same')
    y3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(y3)
    y3 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y3)

    # Channel 1 - Cov Net Layer 7
    x4 = Multiply()([x3, y3])
    x4 = Flatten()(x4)
    x4 = Dense(2048, activation='relu')(x4)
    x4 = Dropout(dropout)(x4)

    # Channel 2 - Cov Net Layer 7
    y4 = Multiply()([x3, y3])
    y4 = Flatten()(y4)
    y4 = Dense(2048, activation='relu')(y4)
    y4 = Dropout(dropout)(y4)

    # Channel 1 - Cov Net Layer 8
    x5 = Multiply()([x4, y4])
    x5 = Dense(2048, activation='relu')(x5)
    x5 = Dropout(dropout)(x5)

    # Channel 2 - Cov Net Layer 8
    y5 = Multiply()([x4, y4])
    y5 = Dense(2048, activation='relu')(y5)
    y5 = Dropout(dropout)(y5)

    # Final Channel - Cov Net 9
    xy = Multiply()([x5, y5])
    xy = Dense(output_dim=num_classes, activation='softmax')(xy)

    return xy, img_input, input_shape, data_format
