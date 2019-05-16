import numpy as np
import copy


class KMeans:
    def __init__(self, class_num, max_iter=10000, epsilon=0.0001):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.class_num = class_num

    def fit(self, X):
        N = X.shape[0]
        dim = X.shape[1]
        belongs = np.zeros(N, dtype=int)
        mu = [np.random.uniform(np.min(X), np.max(X), dim)
              for _ in range(self.class_num)]

        for _ in range(self.max_iter):
            # 一番近い代表点に割り当てる
            for i in range(N):
                belongs[i] = np.argmin([np.linalg.norm(X[i] - mu[k])
                                        for k in range(self.class_num)])
            # 代表点を修正する
            mu_prev = np.array(copy.copy(mu))
            L = [[] for _ in range(self.class_num)]

            for i in range(N):
                L[belongs[i]].append(X[i])
            mu = np.array([np.mean(L[k], axis=0)
                           for k in range(self.class_num)])

            diff = mu - mu_prev

            # 収束判定
            if np.abs(np.sum(diff)) < self.epsilon:
                print("converged")
                break
        self.mu_ = mu
        return self.predict(X)

    def predict(self, X):
        pred = []
        for i in range(len(X)):
            pred.append(np.argmin([np.linalg.norm(X[i] - self.mu_[k])
                                   for k in range(self.class_num)]))
        return np.array(pred)
