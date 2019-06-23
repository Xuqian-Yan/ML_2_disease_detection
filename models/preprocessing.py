from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.feature_selection import VarianceThreshold
import numpy


class Flatten(BaseEstimator, TransformerMixin):
    """Flatten"""
    def __init__(self):
        self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        X = X.reshape(-1, 176, 208, 176)  # Bad practice: hard-coded dimensions
        X1 = X.mean(axis=1)
        X2 = X.mean(axis=2)
        X3 = X.mean(axis=3)
        X1 = X1.reshape(X1.shape[0], -1)
        X2 = X2.reshape(X2.shape[0], -1)
        X3 = X3.reshape(X3.shape[0], -1)
        X_new = numpy.concatenate((X1, X2, X3), axis=1)
        return X_new


class FeatureReduction(BaseEstimator, TransformerMixin):
    """VarianceThreshold"""
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.ind = None

    def fit(self, X, y=None):
        X = check_array(X)
        var = numpy.var(X, axis=0)
        self.ind = numpy.where(var != self.threshold)[0]
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["ind"])
        X = check_array(X)
        X_new = X[:, self.ind]

        return X_new


class Hist(BaseEstimator, TransformerMixin):
    """histogram"""

    def __init__(self, bin=1000):
        self.bin = bin

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        f = VarianceThreshold()
        X = check_array(X)
        X = f.fit_transform(X)
        n = X.shape[0]
        a = numpy.zeros((n, self.bin))
        for i in range(0, n):
            a[i], _ = numpy.histogram(X[i], bins=self.bin, range=(0, 5000))
#        a = f.fit_transform(a)
        X_new = a

        return X_new
