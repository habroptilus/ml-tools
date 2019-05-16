import math
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error


class GaussianKernelNorm1(BaseEstimator, RegressorMixin):
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

    def fit(self, X, y, max_iter=10000, epsilon=0.0001):
        self.x_ = X
        K = self.create_K(X)
        n = K.shape[0]
        theta_next, z_next, u_next = self.init_params(n)
        for i in range(max_iter):
            u_prev = u_next
            z_prev = z_next
            theta_prev = theta_next
            theta_next, z_next, u_next = self.update_params(
                z_prev, u_prev, y, K, n)
            if np.linalg.norm(theta_next - theta_prev, ord=2) < epsilon:
                print(
                    f"iter {i+1} loss {self.loss(theta_next,y,K)} (Converged!)")
                break
            if (i + 1) % 100 == 0:
                print(f"iter {i+1} loss {self.loss(theta_next,y,K)}")
        self.coef_ = theta_next
        return self

    def loss(self, theta, y, K):
        return np.linalg.norm(np.dot(K, theta) - y, ord=2) / 2 + self.c * np.linalg.norm(theta, ord=1)

    def init_params(self, n):
        return np.random.rand(), np.random.rand(), np.random.rand()

    def update_params(self, z_prev, u_prev, y, K, n):
        theta_next = np.linalg.solve(
            np.dot(K, K) + np.eye(n), np.dot(K, y) + z_prev - u_prev)
        z_next = np.maximum(np.zeros(n), theta_next, u_prev - self.c * np.ones(
            n)) + np.minimum(np.zeros(n), theta_next, u_prev + self.c * np.ones(n))
        u_next = u_prev + theta_next - z_next
        return theta_next, z_next, u_next

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
