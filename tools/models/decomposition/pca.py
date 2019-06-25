import numpy as np


class PCA:
    """Pricipal Component Analysis"""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.cov_ = self.get_covariance(X)
        eig_val, eig_vec = np.linalg.eig(self.cov_)
        eig_vec = np.array(eig_vec).T
        index = np.argsort(-eig_val)
        eig_vec = eig_vec[index]
        self.components_ = eig_vec[:self.n_components, :]
        return self.components_

    def transform(self, X):
        return self.components_ @ X

    def get_covariance(self, X):
        z = X - self.mean_
        return np.cov(z[:, 0], z[:, 1], bias=1)
