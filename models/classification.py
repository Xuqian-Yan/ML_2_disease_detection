import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.naive_bayes import GaussianNB
from scipy import stats
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""
    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))


class NaiveBay(BaseEstimator, TransformerMixin):
    '''naive bayes'''

    def __init__(self, fun=None):
        self.fun = fun

    def fit(self, X, y):
        y_new = np.zeros(y.shape[0])
        for i in range(0, y.shape[0]):
            y_new[i] = np.argmax(y[i, :])
        clf = GaussianNB()
        self.fun = clf.fit(X, y_new)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ['fun'])
        probs = self.fun.predict_proba(X)
        return probs

    def score(self, X, y, sample_weight=None):
        y_predicted = self.fun.predict_proba(X)
        scores = np.zeros(y_predicted.shape[0])
        for i in range(0, y_predicted.shape[0]):
            scores[i], _ = stats.spearmanr(y_predicted[i, :], y[i, :])
        score = np.mean(scores)
        print(score)
        return score


class LiSVC(BaseEstimator, TransformerMixin):
    '''SVC with linear kernel'''

    def __init__(self, fun=None):
        self.fun = fun

    def fit(self, X, y):
        y_new = np.zeros(y.shape[0])
        for i in range(0, y.shape[0]):
            y_new[i] = np.argmax(y[i, :])
        clf = SVC(kernel="rbf", probability=True)
        self.fun = clf.fit(X, y_new)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ['fun'])
        probs = self.fun.predict_proba(X)
        return probs

    def score(self, X, y, sample_weight=None):
        y_predicted = self.fun.predict_proba(X)
        scores = np.zeros(y_predicted.shape[0])
        for i in range(0, y_predicted.shape[0]):
            scores[i], _ = stats.spearmanr(y_predicted[i, :], y[i, :])
        score = np.mean(scores)
        print(score)
        return score


class RandomForest(BaseEstimator, TransformerMixin):
    '''randomforest'''

    def __init__(self, fun=None, n_estimators=100):
        self.fun = fun
        self.n_estimators = n_estimators

    def fit(self, X, y):
        y_new = np.zeros(y.shape[0])
        for i in range(0, y.shape[0]):
            y_new[i] = np.argmax(y[i, :])
        clf = RandomForestClassifier(n_estimators=self.n_estimators)
        self.fun = clf.fit(X, y_new)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["fun"])
        probs = self.fun.predict_proba(X)
        return probs

    def score(self, X, y, sample_weight=None):
        y_predicted = self.fun.predict_proba(X)
        scores = np.zeros(y_predicted.shape[0])
        for i in range(0, y_predicted.shape[0]):
            scores[i], _ = stats.spearmanr(y_predicted[i, :], y[i, :])
        score = np.mean(scores)
        print(score)
        return score
