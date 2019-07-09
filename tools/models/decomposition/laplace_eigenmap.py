import numpy as np
from scipy.linalg import eigh


class LaplacianEigenmap:
    def __init__(self, n_components, k):
        """
        param: n_component : embedding dim
        param: k : knn of similarity matrix 
        """
        self.n_components = n_components
        self.k = k

    def transform(self, X):
        W = self.create_similarity_matrix(X)
        D = self.get_degree_matrix(W)
        L = D - W
        eig_val, eig_vec = eigh(L, D)
        index = np.argsort(eig_val)
        eig_vec = eig_vec[index]

    def get_degree_matrix(self, W):
        return np.diag([sum(W[i]) for i in range(len(W))])

    def create_similarity_matrix(self, X):
        """create Similarity matrix (knn)

        :param X: data matrix (data_nX,feature_n)
        """
        W = []
        for x_i in X:
            W.append(self.k_nearest_list(X, x_i))
        return np.array(W)

    def k_nearest_list(self, X, x_i):
        dist_list = [self.dist(x_i, x_j) for x_j in X]
        sorted_list = sorted(dist_list)  # 昇順
        threshold = sorted_list[self.k - 1]
        dist_list = np.array(dist_list)
        return np.where(dist_list <= threshold, 1, 0)

    def dist(self, x_i, x_j):
        return np.dot(x_i - x_j, x_i - x_j)
