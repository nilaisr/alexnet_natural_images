import numpy as np


def princomp(A):
    """ performs principal components analysis
        (PCA) on the n-by-p data matrix A
        Rows of A correspond to observations, columns to variables.

    Returns :
     coeff :
       is a p-by-p matrix, each column containing coefficients
       for one principal component.
     score :
       the principal component scores; that is, the representation
       of A in the principal component space. Rows of SCORE
       correspond to observations, columns to components.
     latent :
       a vector containing the eigenvalues
       of the covariance matrix of A.
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A - np.mean(A.T, axis=1)).T  # subtract the mean (along columns)
    [latent, coeff] = np.linalg.eig(np.cov(M))  # attention:not always sorted
    sortedIdx = np.argsort(-latent)
    latent = latent[sortedIdx]
    explained = 100 * latent / np.sum(latent)
    score = np.dot(coeff.T, M)  # projection of the data in the new space
    coeff = coeff[:, sortedIdx]
    score = score[sortedIdx, :]
    return coeff, score, latent, explained
