from keras.layers import Dense, ZeroPadding2D, Dropout, Flatten, MaxPooling2D, Concatenate, Multiply, Input, merge
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras import backend as K
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
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(filters, (num_row, num_col),
               strides=strides, padding=padding,
               use_bias=False, name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def two_streams():
    # global constants
    num_classes = 444  # number of classes
    dropout = 0.5
    data_format = 'channels_last'

    # # Define image input layer
    input_shape = (227, 227, 3)  # 3 - Number of RGB Colours
    img_input = Input(shape=input_shape)

    # Channel 1 - Conv Net Layer 1
    x = conv2D_bn(img_input, 3, 11, 11, (1, 1), 'same')
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), data_format=data_format)(x)
    x = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x)

    # Channel 2 - Conv Net Layer 1
    y = conv2D_bn(img_input, 3, 11, 11, (1, 1), 'same')
    y = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), data_format=data_format)(y)
    y = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y)

    # Channel 1 - Conv Net Layer 2
    x = conv2D_bn(x, 48, 55, 55, (1, 1), 'same')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(x)
    x = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x)

    # Channel 2 - Conv Net Layer 2
    y = conv2D_bn(y, 48, 55, 55, (1, 1), 'same')
    y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(y)
    y = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y)

    # Channel 1 - Conv Net Layer 3
    x = conv2D_bn(x, 128, 27, 27, (1, 1), 'same')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(x)
    x = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x)

    # Channel 2 - Conv Net Layer 3
    y = conv2D_bn(y, 128, 27, 27, (1, 1), 'same')
    y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(y)
    y = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y)

    # Channel 1 - Conv Net Layer 4
    x1 = Concatenate()([x, y])
    x1 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x1)
    x1 = conv2D_bn(x1, 192, 13, 13, (1, 1), 'same')

    # Channel 2 - Conv Net Layer 4
    y1 = Concatenate()([x, y])
    y1 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y1)
    y1 = conv2D_bn(y1, 192, 13, 13, (1, 1), 'same')

    # Channel 1 - Conv Net Layer 5
    x2 = Concatenate()([x1, y1])
    x2 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x2)
    x2 = conv2D_bn(x2, 192, 13, 13, (1, 1), 'same')

    # Channel 2 - Conv Net Layer 5
    y2 = Concatenate()([x1, y1])
    y2 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y2)
    y2 = conv2D_bn(y2, 192, 13, 13, (1, 1), 'same')

    # Channel 1 - Cov Net Layer 6
    x3 = conv2D_bn(x2, 128, 27, 27, (1, 1), 'same')
    x3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(x3)
    x3 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x3)

    # Channel 2 - Cov Net Layer 6
    y3 = conv2D_bn(y2, 128, 27, 27, (1, 1), 'same')
    y3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format=data_format)(y3)
    y3 = ZeroPadding2D(padding=(1, 1), data_format=data_format)(y3)

    # Channel 1 - Cov Net Layer 7
    print(np.shape(x3), np.shape(y3))
    x3t = np.transpose(x3, (0, 3, 2, 1))
    y3t = np.transpose(y3, (0, 3, 2, 1))
    # x4 = merge([x3, y3], mode='mul', concat_axis=3)
    x4 = Multiply()([x3t, y3t])
    x4 = np.transpose(x4, (2, 1, 0))
    # x4 = Concatenate(axis=3)([x3, y3])
    x4 = Flatten()(x4)
    x4 = Dense(2048, activation='relu')(x4)
    x4 = Dropout(dropout)(x4)

    # Channel 2 - Cov Net Layer 7
    y4 = Multiply()([x3t, y3t])
    y4 = np.transpose(y4, (2, 1, 0))
    # y4 = merge([x3, y3], mode='mul', concat_axis=3)
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
    xy = Dense(output_dim=num_classes,
               activation='softmax')(xy)

    return xy, img_input, input_shape, data_format
