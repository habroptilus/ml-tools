import numpy as np


class GaussianProcessRegressor:
    def __init__(self, kernel, c):
        self.kernel = kernel
        self.c = c

    def fit(self, X, y):
        self.X_train = X
        self.y_mean = np.mean(y)
        y_train = y - self.y_mean
        self.y_train = np.reshape(y_train, (len(y_train), 1))
        self.K = self.get_gram_matrix(X, X, flag=True)

    def predict(self, X):
        mu, _ = self.calc_mu_sigma(X)
        return mu

    def calc_mu_sigma(self, X):
        k_star = self.get_gram_matrix(self.X_train, X, flag=False)
        k_starstar = self.kernel.run(X, X) + self.c
        K_inv = np.linalg.inv(self.K)
        mu = k_star.T @ K_inv @ self.y_train
        sigma = k_starstar - k_star.T@K_inv@k_star
        return mu + self.y_mean, sigma

    def get_gram_matrix(self, A, B, flag):
        """AとBのgram行列を返す.
        return's shape (len(A),len(B))
        flag: クロネッカーつけるかどうか
        """
        return np.array([[self.kernel.run(A[i], B[j]) + self.c * self.kronecker(i, j, flag) for j in range(len(B))] for i in range(len(A))])

    def kronecker(self, i, j, flag):
        return 1 if i == j and flag else 0
