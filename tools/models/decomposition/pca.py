import numpy as np


class PCA:
    """Pricipal Component Analysis"""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.cov_ = self.get_covariance(X)
        W, v = np.linalg.eig(self.cov_)
        v = np.array(v).T
        self.components_ = v[:self.n_components, :]
        return self.components_

    def transform(self, X):
        return self.components_ @ X

    def get_covariance(self, X):
        z = X - self.mean_
        return np.cov(z[:, 0], z[:, 1], bias=1)
