from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.utils.validation import check_is_fitted
import math


class GaussianKernelLSPClassifier:
    def __init__(self, h, c, class_num):
        """initialize
        :param h: gauss window 
        :param c: reglarizer
        :param class_num: the number of class
        """
        self.h = h
        self.c = c
        self.class_num = class_num

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
        theta_list = []
        for c in range(self.class_num):
            pi_y = self.get_pi_y(y, c)
            theta_y = np.linalg.solve(
                np.dot(K.T, K) + self.c * np.eye(n), np.dot(K.T, pi_y))
            theta_list.append(theta_y)
        self.coef_ = theta_list
        return self

    def get_pi_y(self, y, c):
        return np.where(y == c, 1, 0)

    def score(self, X):
        check_is_fitted(self, 'coef_')
        check_is_fitted(self, 'x_')
        scores = []
        for x_i in X:
            k_i = np.array([self.k(x_i, x_j) for x_j in self.x_])
            scores.append([np.dot(coef, k_i) for coef in self.coef_])
        return scores

    def predict(self, X):
        scores = self.score(X)
        return np.argmax(scores, axis=1)

    def evaluate(self, X, y):
        pred_y = self.predict(X)
        return accuracy_score(y, pred_y)

    def predict_proba(self, X):
        scores = self.score(X)
        return np.array([np.maximum(0, score) / np.sum(np.maximum(0, score)) for score in scores])
