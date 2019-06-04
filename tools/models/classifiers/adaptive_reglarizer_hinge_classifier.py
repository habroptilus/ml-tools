import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score


class AdaptiveReglarizerHingeClassifier:
    """線形モデル + 二乗ヒンジ誤差 + 適応正則化model.
    ラベルは正例が1,負例は-1
    オンライン学習
    """

    def __init__(self, gamma, seed=None):
        self.gamma = gamma
        self.seed = seed

    def fit(self, X, y, epochs, batch_size):
        X = self.add_const(X)
        N = X.shape[0]
        batch_num = N // batch_size
        mu, sigma = self.init_params(X.shape[1])
        for epoch in range(epochs):
            if (epoch + 1) % 100 == 0:
                print(f"epoch {epoch+1}")
            for batch in range(batch_num):
                X_batched = X[batch * batch_size:(batch + 1) * batch_size, :]
                y_batched = y[batch * batch_size:(batch + 1) * batch_size]
                sigma = self.update_sigma(sigma, X_batched, y_batched)
                mu = self.update_mu(mu, sigma, X_batched, y_batched)
            if N % batch_size != 0:
                X_batched = X[batch_num * batch_size:, :]
                y_batched = y[batch_num * batch_size:]
                sigma = self.update_sigma(sigma, X_batched, y_batched)
                mu = self.update_mu(mu, sigma, X_batched, y_batched)
        self.coef_ = mu
        return

    def init_params(self, d):
        """入力次元d"""
        np.random.seed(self.seed)
        mu = np.random.randn(d)
        a = np.random.uniform(-1, 1, (d, d))
        sigma = np.dot(a, a.T)
        return np.reshape(mu, (d, 1)), sigma

    def update_mu(self, prev_mu, prev_sigma, X, y):
        update_vecs = []
        for i in range(X.shape[0]):
            x = np.reshape(X[i], (len(X[i]), 1))
            bunsi = y[i] * max(0, 1.0 - y[i] * prev_mu.T @ x)
            bunbo = x.T @ prev_sigma @ x + self.gamma
            update_vecs.append((bunsi / bunbo) * (prev_sigma @ x))
        return prev_mu + np.mean(np.array(update_vecs), axis=0)

    def update_sigma(self, prev_sigma, X, y):
        update_mats = []
        for i in range(X.shape[0]):
            x = np.reshape(X[i], (len(X[i]), 1))
            bunsi = prev_sigma@x @ x.T @ prev_sigma
            bunbo = x.T @ prev_sigma @ x
            update_mats.append(bunsi / bunbo)
        return prev_sigma - np.mean(np.array(update_mats), axis=0)

    def predict(self, X):
        """return 1 or -1"""
        check_is_fitted(self, 'coef_')
        scores = self.add_const(X)@ self.coef_
        return np.where(scores.flatten() > 0, 1, -1)

    def evaluate(self, X, y):
        #y[y == -1] = 0
        pred_y = self.predict(X)
        #pred_y[pred_y == -1] = 0
        return accuracy_score(y, pred_y)

    def add_const(self, X):
        """add const to X"""
        const = np.ones((len(X), 1))
        return np.hstack((const, X))
