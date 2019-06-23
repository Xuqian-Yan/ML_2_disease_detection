from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
import numpy as np


class NonZeroSelection(BaseEstimator, TransformerMixin):
    """Select non-zero voxels"""
    def fit(self, X, y=None):
        X = check_array(X)
        self.nonzero = X.sum(axis=0) > 0
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["nonzero"])
        X = check_array(X)
        return X[:, self.nonzero]


class RandomSelection(BaseEstimator, TransformerMixin):
    """Random Selection of features"""
    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
                            n_features,
                            self.n_components,
                            random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new


class LayerHist (BaseEstimator, TransformerMixin):
    def __init__(self, bins=50):
        self.bins = bins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X_reshaped = np.reshape(X, (-1, 176, 208, 176))
        n = X_reshaped.shape[0]
        a = np.zeros((n, 8*8*self.bins))
        b = np.zeros((8, 8*self.bins))
        for i in range(0, n):
            for j in range(0, 8):  # 176 =8 *22
                for k in range(0, 8):  # 208 = 8 * 26
                    b[j, range(k * self.bins, (k + 1) * self.bins)] = (
                        np.histogram(X_reshaped[i, (j * 22):((j + 1) * 22),
                                     (k * 26):((k + 1) * 26), :],
                                     bins=self.bins,
                                     range=(100, 5000))[0]
                    )
            a[i, :] = np.reshape(b, (1, -1))
        return a


class CubeHist (BaseEstimator, TransformerMixin):
    def __init__(self, bins=50):
        self.bins = bins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X_reshaped = np.reshape(X, (-1, 176, 208, 176))
        n = X_reshaped.shape[0]
        a = np.zeros((n, 8*8*8*self.bins))
        b = np.zeros((8, 8*8*self.bins))
        c = np.zeros((8, 8*self.bins))
        for i in range(0, n):
            for j in range(0, 8):  # 176 =8 *22
                for k in range(0, 8):  # 208 = 8 * 26
                    for w in range(0, 8):  # 176 =8 *22
                        c[k, range(w * self.bins, (w + 1) * self.bins)] = (
                            np.histogram(X_reshaped[i, (j * 22):((j + 1) * 22),
                                         (k * 26):((k + 1) * 26),
                                         (w * 22):((w + 1) * 22)],
                                         bins=self.bins,
                                         range=(100, 4500))[0]
                        )
                b[j, :] = np.reshape(c, (1, -1))
            a[i, :] = np.reshape(b, (1, -1))
        return a
