from sklearn.metrics import accuracy_score
import numpy as np
from itertools import combinations


class OneVsOneClassifier:
    """一対一分類器."""

    def __init__(self, class_num, classifier, params=None):
        """binary classifierを受け取り、c(c+1)/2個のインスタンスを作成する."""
        self.class_num = class_num
        self.perm = list(combinations(list(range(class_num)), 2))
        if params is None:
            self.classifiers = [classifier() for _ in range(len(self.perm))]
        else:
            self.classifiers = [classifier(**params)
                                for _ in range(len(self.perm))]

    def fit(self, X, y):
        for i in range(len(self.classifiers)):
            X_i, y_i = self.extract_dataset(X, y, i)
            print(f"Training classifier{i}...")
            self.classifiers[i].fit(X_i, y_i)
        return self

    def extract_dataset(self, X, y, i):
        pos = self.perm[i][0]
        neg = self.perm[i][1]
        X = X[(y == pos) | (y == neg)]
        y = y[(y == pos) | (y == neg)]
        y = np.where(y == pos, 1, -1)
        return X, y

    def predict(self, X):
        votes = np.zeros((len(X), self.class_num))
        for i in range(len(self.classifiers)):
            prediction = self.classifiers[i].predict(X)
            voted = np.where(prediction == 1, self.perm[i][0], self.perm[i][1])
            for j in range(len(voted)):
                votes[j, voted[j]] += 1
        return np.argmax(votes, axis=1)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)
