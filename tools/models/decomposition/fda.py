import numpy as np
from scipy.linalg import eigh


class FDA:
    """Fisher Discriminant Analysis."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y):
        X = X - np.mean(X, axis=0)
        S_w, S_b = self.create_W(X, y)
        eig_val, eig_vec = eigh(S_b, S_w)
        index = np.argsort(eig_val)
        eig_vec = eig_vec[index]
        components = []
        for vec in reversed(eig_vec):  # normalize
            normalized_vec = vec / np.linalg.norm(vec)
            components.append(normalized_vec)
        self.components_ = np.array(components[:self.n_components])

    def transform(self, X):
        return self.components_ @ X

    def create_between_matrix(self, X, y):
        S_b = np.zeros((X.shape[1], X.shape[1]))
        for c in np.unique(y):
            x = X[y == c]
            n_y = len(x)
            mu_y = np.mean(x, axis=0)
            mu_y = np.reshape(mu_y, (len(mu_y), 1))
            S_b = S_b + n_y * mu_y@mu_y.T
        return S_b

    def create_W(self, X, y):
        S_b = self.create_between_matrix(X, y)
        C = self.create_C(X)
        S_w = C - S_b
        return S_w, S_b

    def create_C(self, X):
        return X@X.T
