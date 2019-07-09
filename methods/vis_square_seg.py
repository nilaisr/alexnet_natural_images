import matplotlib as plt
import numpy as np


def vis_square_seg(data, name_fig, index_lum_ker):
    # Function that displays kernels of the first layer as images. kernels classified as color agnostic are surrounded by black.
    # Take an array of shape (n, height, width) or (n, height, width, 3)
    # and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    gp_top = data[0:48]
    gp_bot = data[48:]
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, 0), (1, 2), (1, 2)) + ((0, 0),) * (data.ndim - 3))
    gp_top = np.pad(gp_top, padding, mode='constant', constant_values=1)  # pad with ones (white)
    gp_bot = np.pad(gp_bot, padding, mode='constant', constant_values=1)  # pad with ones (white)
    data = np.concatenate((gp_top, gp_bot), axis=0)
    for i in index_lum_ker:
        if i < 48:
            data[i, :-1, (0, -2), :] = 0
            data[i, (0, -2), :-1, :] = 0
        else:
            data[i, :-1, (0, -2), :] = 0
            data[i, (0, -2), :-1, :] = 0
    # tile the filters into an image
    data = data.reshape((12, 8) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((12 * data.shape[1], 8 * data.shape[3]) + data.shape[4:])
    data = np.insert(data, data.shape[0] / 2, np.ones((3, data.shape[1], 3)), 0)
    data = data[:-1, :-1]
    # print data.shape
    plt.imshow(data)
    plt.show()
    plt.imsave(name_fig + '.png', data)
    plt.imsave(name_fig + '.eps', data)
    return data
