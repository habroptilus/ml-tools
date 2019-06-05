import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score


class PAHingeClassifier:
    """Passive Aggressive learning model.
    線形モデル + hinge loss
    """

    def __init__(self, gamma, seed=None):
        self.gamma = gamma
        self.seed = seed

    def fit(self, X, y, epochs, batch_size):
        X = self.add_const(X)
        N = X.shape[0]
        batch_num = N // batch_size
        theta = self.init_theta(X.shape[1])
        for epoch in range(epochs):
            if (epoch + 1) % 100 == 0:
                print(f"epoch {epoch+1}")
            for batch in range(batch_num):
                X_batched = X[batch * batch_size:(batch + 1) * batch_size, :]
                y_batched = y[batch * batch_size:(batch + 1) * batch_size]
                theta = self.update_theta(X_batched, y_batched, theta)
            if N % batch_size != 0:
                X_batched = X[batch_num * batch_size:, :]
                y_batched = y[batch_num * batch_size:]
                theta = self.update_theta(X_batched, y_batched, theta)
        self.coef_ = theta
        return

    def update_theta(self, X, y, theta):
        updates = []
        for i in range(X.shape[0]):
            x = np.reshape(X[i], (len(X[i]), 1))
            delta_theta = (y[i] * max(0, 1 - y[i] * theta.T@x) /
                           (np.linalg.norm(x)**2 + self.gamma)) * x
            updates.append(delta_theta)
        return theta + np.mean(np.array(updates), axis=0)

    def init_theta(self, d):
        """入力次元d"""
        np.random.seed(self.seed)
        theta = np.random.randn(d)
        return np.reshape(theta, (d, 1))

    def predict(self, X):
        """return 1 or -1"""
        check_is_fitted(self, 'coef_')
        scores = self.add_const(X)@ self.coef_
        return np.where(scores.flatten() > 0, 1, -1)

    def evaluate(self, X, y):
        pred_y = self.predict(X)
        return accuracy_score(y, pred_y)

    def add_const(self, X):
        """add const to X"""
        const = np.ones((len(X), 1))
        return np.hstack((const, X))
