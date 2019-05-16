from sklearn.metrics import accuracy_score
import numpy as np


class OneVsRestClassifier:
    """一対他分類器."""

    def __init__(self, class_num, classifier, params):
        """binary classifierを受け取り、クラスの数分インスタンスを作成する."""
        self.class_num = class_num
        self.classifiers = [classifier(**params) for _ in range(class_num)]

    def fit(self, X, y):
        for i in range(self.class_num):
            print(f"Training classifier{i}...")
            self.classifiers[i].fit(X, self.re_labaling(y, i))
        return self

    def re_labaling(self, y, pos_label):
        """labelを受け取り、pos_labelに指定したカテゴリを+1、それ以外を-1にラベリングしなおしたデータを返す."""
        return np.where(y == pos_label, 1, -1)

    def predict(self, X):
        scores = np.array([model.score(X) for model in self.classifiers])
        return np.argmax(scores, axis=0)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)
