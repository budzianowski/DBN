import numpy as np
import sklearn.preprocessing


def random_columns(x, n_select):
    """

    :rtype: object
    """
    n_features = x.shape[0]
    n_columns = x.shape[1]
    perm = np.arange(n_columns)
    np.random.shuffle(perm)
    y = x[:, perm[0:n_select]]  # TODO: check how many numbers

    return y, perm


def normalize_samples(x):
    samples = x.shape[1]
    for ii in range(samples):
        s = x[:, ii]
        mins = min(s)
        maxs = max(s)
        rans = maxs-mins
        x[:, ii] = (s-mins)/rans

    return x


def binarize(x, threshold=0.0):
    sklearn.preprocessing.binarize(x, threshold, copy=False)
    return x

