import numpy as np
import math
from scipy.linalg import eigh


class LPP:
    """Locality Preserving Projection."""

    def __init__(self, n_components, h=1 / np.sqrt(2)):
        self.n_components = n_components
        self.h = h

    def fit(self, X):
        X = X - np.mean(X, axis=0)
        W = self.create_similarity_matrix(X)
        D = self.get_degree_matrix(W)
        L = D - W
        A = X.T @ L @ X
        B = X.T @ D @ X
        eig_val, eig_vec = eigh(A, B)
        components = []
        for vec in eig_vec:  # normalize
            normalized_vec = vec / np.linalg.norm(vec)
            components.append(normalized_vec)
        self.components_ = np.array(components[:self.n_components])
        return self.components_

    def gaussian_kernel(self, x_i, x_j):
        """kernel function"""
        return math.e**(-np.dot(x_i - x_j, x_i - x_j) / (2 * self.h**2))

    def get_degree_matrix(self, W):
        return np.diag([sum(W[i]) for i in range(len(W))])

    def create_similarity_matrix(self, X):
        """create Similarity matrix

        :param X: data matrix (data_nX,feature_n)
        """
        W = []
        for x_i in X:
            K_i = []
            for x_j in X:
                K_i.append(self.gaussian_kernel(x_i, x_j))
            W.append(K_i)
        return np.array(W)
