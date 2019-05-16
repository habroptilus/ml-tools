import numpy as np


class TukeyRegression:
    def __init__(self, eta=3, epsilon=0.00001, iter_num=10000, seed=None):
        self.eta = eta
        self.epsilon = epsilon
        self.seed = seed
        self.theta = np.random.randn(2)
        self.iter_num = iter_num

    def fit(self, x, y):
        np.random.seed(self.seed)
        const = np.ones(len(x))
        Phi = np.stack([const, x], axis=1)

        for i in range(self.iter_num):
            W = self.get_W(Phi, y)
            theta = np.linalg.solve(
                np.dot(np.dot(Phi.T, W), Phi), np.dot(np.dot(Phi.T, W), y))
            if np.linalg.norm(theta - self.theta) < self.epsilon:
                print("Converged!")
                self.theta = theta
                break
            self.theta = theta
            print(f"iter {i+1} loss {self.loss(Phi,y)}")

    def predict(self, x):
        const = np.ones(len(x))
        Phi = np.stack([const, x], axis=1)
        return self.f(Phi)

    def f(self, Phi):
        return np.dot(Phi, self.theta)

    def W(self, r):
        w = 1 - r / self.eta**2
        w[abs(r) >= self.eta] = 0
        return np.diag(w)

    def get_W(self, Phi, y):
        y_hat = self.f(Phi)
        return self.W(y_hat - y)

    def loss(self, Phi, y):
        return np.linalg.norm(self.f(Phi) - y) / 2
