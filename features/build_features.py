from functools import wraps

import numpy as np
from sklearn.model_selection import train_test_split

from config import M, T, df


def feature(func):
    """
    A decorator to add the feature to the list of features to be computed
    WARNING: The decorated function must take (xi, mu) as an input
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.is_feature = True
    return wrapper


@feature
def quantile(xi, mu):
    """
    Prepare new features for a better spacial representation of the data.

    """
    # TODO: compute the quantiles of each time series
    # Idea: the Student distribution has lots of info given from its quantiles (tails, etc.)
    xi_quantile = np.zeros((M, len(mu), 3))
    for i in range(M):
        for j in range(len(mu)):
            xi_quantile[i, j, :] = np.quantile(xi[i, :, j], [0.25, 0.5, 0.75])

    return xi_quantile


def prepare_train(xi, mu, test_size=0.2, compute_features=False):
    """
    Apply all relevant feature engineering functions
    as well as train/test split.

    Target is mu for any timeseries xi.
    """
    # a feature has wrapper @feature which is a decorator that adds
    # the feature to the list of features to be computed
    features = []
    for name in dir():
        obj = eval(name)
        if callable(obj) and getattr(obj, 'is_feature', False):
            features.append(obj)

    if compute_features:
        # compute all features
        X = np.zeros((M, len(features)))
        for i in range(M):
            for j, f in enumerate(features):
                X[i, j] = f(xi[i], mu)
        y = mu
    else:
        # INFO: We are trying to predict mu from xi!
        X = xi.reshape(M * len(mu), T)
        y = np.zeros((M * len(mu), 1))
        assert X.shape[0] == M * \
            len(mu), "Something is wrong in your training data in terms of shape"

    # split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test
