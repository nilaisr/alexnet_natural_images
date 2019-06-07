import numpy as np


# Function that converts RGBPCA to RGB.
def pca2rgb(x):
    # The following matrix are the exact PCs from the Imagenet 2012 pixels color distribution.
    m = np.array([[0.65, 1.02, -0.49], [0.67, 0.004, 1], [0.67, -0.98, -0.51]])
    return np.dot(x, np.linalg.inv(m))
