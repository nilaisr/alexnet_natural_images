import numpy as np


# Function that converts RGB to RGBPCA.
def rgb2pca(x):
    # The following matrix are the exact PCs from the Imagenet 2012 pixels color distribution.
    m = np.array([[0.65, 1.02, -0.49], [0.67, 0.004, 1], [0.67, -0.98, -0.51]])
    x = x.astype('float64')
    x_pca = np.zeros((224, 224, 3))
    for i in range(x.shape[0]):
        for k in range(m.shape[0]):
            for j in range(x.shape[1]):
                x_pca[i, j, :] = sum(x[i, j, :] * m[k, :])
    x_pca = np.around(x_pca)
    return x_pca
