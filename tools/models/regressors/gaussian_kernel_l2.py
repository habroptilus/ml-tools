import math
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error


class GaussianKernelNorm2(BaseEstimator, RegressorMixin):
    """ガウスカーネルによる最小二乗回帰(l2正則化)"""

    def __init__(self, h, c):
        """initialize
        :param h: ガウス幅
        :param c: 正則化係数
        """
        self.h = h
        self.c = c

    def k(self, x_i, x_j):
        """kernel function"""
        return math.e**(-np.dot(x_i - x_j, x_i - x_j) / (2 * self.h**2))

    def create_K(self, X):
        """create Kernel matrix

        :param X: data matrix (data_n,feature_n)
        :return K: kernel matrix (data_n,data_n)
        """
        K = []
        for x_i in X:
            K_i = []
            for x_j in X:
                K_i.append(self.k(x_i, x_j))
            K.append(K_i)
        return np.array(K)

    def fit(self, X, y):
        self.x_ = X
        K = self.create_K(X)
        n = K.shape[0]
        self.coef_ = np.linalg.solve(
            np.dot(K, K) + self.c * np.eye(n), np.dot(K.T, y))
        return self

    def predict(self, X):
        check_is_fitted(self, 'coef_')
        check_is_fitted(self, 'x_')
        pred_y = []
        for x_i in X:
            k_i = np.array([self.k(x_i, x_j) for x_j in self.x_])
            pred_y.append(np.dot(self.coef_, k_i))
        return np.array(pred_y)

    def evaluate(self, X, y):
        pred_y = self.predict(X)
        return mean_squared_error(pred_y, y)
