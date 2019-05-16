import pandas as pd
import numpy as np


class DigitDataLoader:
    def __init__(self, class_num=10, shuffle=True):
        self.shuffle = shuffle
        self.class_num = class_num

    def _load(self, kind):
        X = []
        Y = []
        for label in range(self.class_num):
            df = pd.read_csv(f"../data/digit/digit_{kind}{label}.csv")
            x = list(df.values)
            y = [label] * len(df)
            X += x
            Y += y
        X, Y = np.array(X), np.array(Y)
        if self.shuffle:
            p = np.random.permutation(len(X))
            X = X[p]
            Y = Y[p]
        return X, Y

    def load(self):
        X_train, y_train = self._load("train")
        X_test, y_test = self._load("test")
        return X_train, y_train, X_test, y_test
