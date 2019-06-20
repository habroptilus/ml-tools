from sklearn.linear_model import LinearRegression
import numpy as np


class LinearLSClassifier:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        theta = np.array([self.model.intercept_] + list(self.model.coef_))
        self.theta = np.reshape(theta, (len(theta), 1))

    def predict(self, X):
        scores = self.model.predict(X)
        return np.where(scores > 0, 1, -1)
