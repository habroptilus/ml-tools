import numpy as np


class ClassWeightedLinearClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X_train = self.add_const(X)
        self.y_train = y
        self.pi_train = sum(y[y == 1]) / len(y)
        self.A11 = self.get_A(1, 1)
        self.A10 = self.get_A(1, -1)
        self.A00 = self.get_A(-1, -1)
        return

    def get_A(self, y1, y2):
        """A[y1,y2]を返す."""
        X1 = self.X_train[self.y_train == y1]
        X2 = self.X_train[self.y_train == y2]
        return self.get_sample_mean(X1, X2)

    def get_b(self, X_test, y):
        """b_yを返す."""
        X_train_y = self.X_train[self.y_train == y]
        return self.get_sample_mean(X_train_y, X_test)

    def get_sample_mean(self, A, B):
        result = []
        for a in A:
            for b in B:
                result.append(np.linalg.norm(a - b))
        return np.mean(result)

    def predict(self, X):
        X_test = self.add_const(X)
        y = np.reshape(self.y_train, (len(self.y_train), 1))
        P = self.get_P(X_test)
        A = self.X_train.T@P@self.X_train
        b = self.X_train.T@P@y
        self.theta = np.linalg.solve(A, b)
        scores = X_test @ self.theta
        return np.where(scores > 0, 1, -1)

    def get_P(self, X_test):
        pi = self.get_pi(X_test)
        print(f"estimated pi: {pi:.3f}")
        p = np.array([pi / self.pi_train if y == 1 else (1 - pi) /
                      (1 - self.pi_train) for y in self.y_train])
        return np.diag(p)

    def get_pi(self, X_test):
        b1 = self.get_b(X_test, 1)
        b0 = self.get_b(X_test, -1)
        pi_tilde = (self.A10 - self.A00 - b1 + b0) / \
            (2 * self.A10 - self.A11 - self.A00)
        return min(1, max(0, pi_tilde))

    def add_const(self, X):
        """add const to X"""
        const = np.ones((len(X), 1))
        return np.hstack((const, X))
