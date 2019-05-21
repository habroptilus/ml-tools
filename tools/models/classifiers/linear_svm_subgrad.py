import numpy as np
from sklearn.utils.validation import check_is_fitted


class LinearSVMsubgrad():
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
                print("Converged!")
                theta = theta_next
                break
            theta = theta_next
            if (i + 1) % 100 == 0:
                print(f"iter {i+1}")
        self.coef_ = theta

    def grad(self, phi, y, theta, K):
        return self.subgrad(phi, y, theta) + self.lam * np.dot(K, theta)

    def subgrad(self, phi, y, theta):
        scores = 1 - np.dot(phi, theta) * y
        s_grad = -sum((phi * y)[scores > 0])
        assert len(s_grad) == len(theta), s_grad.shape
        return s_grad

    def get_K(self, phi):
        """calc products for K."""
        return np.dot(phi, phi.T)

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
        const = np.ones(len(X))
        return np.stack([const, X], axis=1)
