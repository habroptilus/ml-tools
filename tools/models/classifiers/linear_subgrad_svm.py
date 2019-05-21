import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score


class LinearSubgradSVM():
    """線形モデルにサポートベクターマシンの劣勾配アルゴリズムを適用したモデル."""

    def __init__(self, lam, lr, max_iter, epsilon):
        """
        :param lam: regralizer
        :param lr: learning rate
        :param max_iter: the number of max iteration
        :param epsilon: threshold used for checking convergence
        """
        self.lam = lam
        self.lr = lr
        self.max_iter = max_iter
        self.epsilon = epsilon

    def fit(self, X, y):
        phi = self.add_const(X)
        theta = self.init_theta(phi.shape[1])
        K = self.get_K(phi)
        for i in range(self.max_iter):
            theta_next = theta - self.lr * self.grad(phi, y, theta, K)
            if np.linalg.norm(theta_next - theta) < self.epsilon:
                print(f"Converged! loss {self.loss(phi,y,theta,K)}")
                theta = theta_next
                break
            theta = theta_next
            if (i + 1) % 1000 == 0:
                print(f"iter {i+1} loss {self.loss(phi,y,theta,K)}")
        self.coef_ = theta

    def grad(self, phi, y, theta, K):
        return self.subgrad(phi, y, theta) + self.lam * np.dot(K, theta)

    def subgrad(self, phi, y, theta):
        scores = 1 - np.dot(phi, theta) * y
        y = y.reshape((len(y), 1))
        if sum(scores) > 0:
            s_grad = -sum((y * phi)[scores > 0])
        else:
            s_grad = np.zeros(len(theta))
        assert len(s_grad) == len(theta), s_grad.shape
        return s_grad

    def loss(self, phi, y, theta, K):
        return sum(np.maximum(0, 1 - np.dot(phi, theta) * y)) + \
            self.lam * np.dot(np.dot(theta.T, K), theta) / 2

    def get_K(self, phi):
        """calc products for K."""
        return np.dot(phi.T, phi)

    def init_theta(self, features_n):
        """initialize theta.
        :params features_n: the number of features.(including constant term)
        """
        return np.random.randn(features_n)

    def predict(self, X):
        check_is_fitted(self, 'coef_')
        scores = np.dot(self.add_const(X), self.coef_)
        return np.where(scores > 0, 1, -1)

    def add_const(self, X):
        """add const to X"""
        const = np.ones((len(X), 1))
        return np.hstack((const, X))

    def evaluate(self, X, y):
        pred_y = self.predict(X)
        return accuracy_score(y, pred_y)
