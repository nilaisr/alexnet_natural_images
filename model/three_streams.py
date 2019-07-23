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
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    bn_axis = 3
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu', name=name)(x)
    return x


def three_streams():
    num_classes = 35  # number of classes
    dropout = 0.5

    data_format = 'channels_last'
    input_shape = (224, 224, 3)  # 3 - Number of RGB Colours

    img_input = Input(shape=input_shape)

    # Channel 1 - Conv Net Layer 1
    x = conv2D_bn(img_input, 32, 11, 11, strides=(4, 4), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(x)

    # Channel 2 - Conv Net Layer 1
    y = conv2D_bn(img_input, 32, 11, 11, strides=(4, 4), padding='same')
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(y)

    # Channel 3 - Conv Net Layer 1
    z = conv2D_bn(img_input, 32, 11, 11, strides=(4, 4), padding='same')
    z = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(z)

    # Channel 1 - Conv Net Layer 2
    x = conv2D_bn(x, 84, 5, 5, strides=(1, 1), padding='valid')
    x = ZeroPadding2D(padding=(2, 2), data_format=data_format)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(x)

    # Channel 2 - Conv Net Layer 2
    y = conv2D_bn(y, 84, 5, 5, strides=(1, 1), padding='valid')
    y = ZeroPadding2D(padding=(2, 2), data_format=data_format)(y)
    y = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(y)

    # Channel 3 - Conv Net Layer 2
    z = conv2D_bn(z, 84, 5, 5, strides=(1, 1), padding='valid')
    z = ZeroPadding2D(padding=(2, 2), data_format=data_format)(z)
    z = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(z)

    # Channel 1 - Conv Net Layer 3
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)

    # Channel 2 - Conv Net Layer 3
    y = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(y)

    # Channel 3 - Conv Net Layer 3
    z = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(z)

    # Channel 1 - Conv Net Layer 4
    x1 = Concatenate()([x, y, z])
    x1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x1)

    # Channel 2 - Conv Net Layer 4
    y1 = Concatenate()([x, y, z])
    y1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(y1)

    # Channel 3 - Conv Net Layer 4
    z1 = Concatenate()([x, y, z])
    z1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(z1)

    # Channel 1 - Conv Net Layer 5
    x2 = Concatenate()([x1, y1, z1])
    x2 = Conv2D(86, (3, 3), strides=(1, 1), padding='same')(x2)
    x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(x2)

    # Channel 2 - Conv Net Layer 5
    y2 = Concatenate()([x1, y1, z1])
    y2 = Conv2D(86, (3, 3), strides=(1, 1), padding='same')(y2)
    y2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(y2)

    # Channel 3 - Conv Net Layer 5
    z2 = Concatenate()([x1, y1, z1])
    z2 = Conv2D(86, (3, 3), strides=(1, 1), padding='same')(z2)
    z2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=data_format)(z2)

    # # Channel 1 - Cov Net Layer 6
    x3 = Multiply()([x2, y2, z2])
    x3 = Flatten()(x3)
    x3 = Dense(1366, activation='relu')(x3)
    x3 = Dropout(dropout)(x3)

    # Channel 2 - Cov Net Layer 6
    y3 = Multiply()([x2, y2, z2])
    y3 = Flatten()(y3)
    y3 = Dense(1366, activation='relu')(y3)
    y3 = Dropout(dropout)(y3)

    # Channel 3 - Cov Net Layer 6
    z3 = Multiply()([x2, y2, z2])
    z3 = Flatten()(z3)
    z3 = Dense(1366, activation='relu')(z3)
    z3 = Dropout(dropout)(z3)

    # Channel 1 - Cov Net Layer 7
    x4 = Multiply()([x3, y3, z3])
    x4 = Dense(1366, activation='relu')(x4)
    x4 = Dropout(dropout)(x4)

    # Channel 2 - Cov Net Layer 7
    y4 = Multiply()([x3, y3, z3])
    y4 = Dense(1366, activation='relu')(y4)
    y4 = Dropout(dropout)(y4)

    # Channel 3 - Cov Net Layer 7
    z4 = Multiply()([x3, y3, z3])
    z4 = Dense(1366, activation='relu')(z4)
    z4 = Dropout(dropout)(z4)

    # Final Channel - Cov Net 9
    xyz = Multiply()([x4, y4, z4])
    output = Dense(num_classes, activation='softmax')(xyz)

    return output, img_input, input_shape
