import numpy as np


class GaussianProcessRegressor:
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.K = self.get_gram_matrix(X, X)

    def predict(self, X):
        mu, _ = self.calc_mu_sigma(X)
        return mu

    def calc_mu_sigma(self, X, y):
        y = np.reshape(y, (len(y), 1))
        k_star = self.get_gram_matrix(self.X_train, X)
        k_starstar = self.kernel.run(X, X)
        K_inv = np.linalg.inv(self.K)
        mu = k_star.T @ K_inv @ y
        sigma = k_starstar - k_star.T@K_inv@k_star
        return mu, sigma

    def get_gram_matrix(self, A, B):
        """AとBのgram行列を返す.
        return's shape (len(A),len(B))
        """
        return np.array([[self.kernel.run(A[i], B[j]) for j in range(len(B))] for i in range(len(A))])
