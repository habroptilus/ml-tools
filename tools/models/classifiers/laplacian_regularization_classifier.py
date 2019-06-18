import numpy as np
import math
from sklearn.utils.validation import check_is_fitted


class LaplacianRegularizationClassifier:
    """ラプラス正則化ガウスカーネルモデルで最小二乗分類.
    近傍グラフはガウスカーネルを用いる.
    label : {+1,-1,None (unlabeled)}
    """

    def __init__(self, lam, nu, h_w, h_m):
        """Intialization.

        :params lam: paramの正則化係数
        :params nu: laplace正則化の係数
        :params h_w: kernel window for W (neighborhood matrix)
        :params h_m: kernel window for K (gram matrix)
        """
        self.lam = lam
        self.nu = nu
        self.h_w = h_w
        self.h_m = h_m

    def fit(self, X, y):
        self.X_train = X
        X_labeled = X[(y == 1) | (y == -1)]
        y = y[(y == 1) | (y == -1)]
        y = np.reshape(y, (len(y), 1))
        y = y.astype(float)  # Noneだったからobjectになってるはずなので直す
        data_n = X.shape[0]
        K_tilde = self.create_kernel_matrix(X_labeled, X, self.h_m)
        K = self.create_kernel_matrix(X, X, self.h_m)
        W = self.create_kernel_matrix(X, X, self.h_w)
        L = self.get_laplacian(W)
        A = K_tilde.T@K_tilde + self.lam * \
            np.eye(data_n) + 2 * self.nu * K.T@L@K
        b = K_tilde.T @ y
        theta = np.linalg.solve(A, b)
        self.coef_ = theta
        return

    def get_laplacian(self, W):
        D = self.get_degree_matrix(W)
        return D - W

    def get_degree_matrix(self, W):
        return np.diag([sum(W[i]) for i in range(len(W))])

    def create_kernel_matrix(self, X, Y, h):
        """create Kernel matrix

        :param X: data matrix (data_nX,feature_n)
        :param Y: data matrix (data_nY,feature_n)
        :return K or W: kernel matrix (data_nX,data_nY)
        """
        K = []
        for x in X:
            K_i = []
            for y in Y:
                K_i.append(self.kernel(x, y, h))
            K.append(K_i)
        return np.array(K)

    def kernel(self, x_i, x_j, h):
        """kernel function"""
        return math.e**(-np.dot(x_i - x_j, x_i - x_j) / (2 * h**2))

    def predict(self, X):
        """return 1 or -1"""
        check_is_fitted(self, 'coef_')
        K = self.create_kernel_matrix(X, self.X_train, self.h_m)
        scores = K @ self.coef_
        return np.where(scores.flatten() > 0, 1, -1)
