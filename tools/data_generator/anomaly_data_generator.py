"""外れ値を含む人工データ生成クラス."""
import numpy as np


class AnomalyDataGenerator:
    def __init__(self, theta1, theta2, outlier_num=5, outlier=15, start=-3, stop=3, seed=None):
        self.seed = seed
        self.stop = stop
        self.start = start
        self.theta1 = theta1
        self.theta2 = theta2
        self.outlier = outlier
        self.outlier_num = outlier_num

    def run(self, n):
        """データ生成.

        :params n:data num
        """
        np.random.seed(self.seed)
        x = np.linspace(self.start, self.stop, n)
        epsilon = 2 * np.random.randn(n)
        y = self.theta1 * x + self.theta2 + epsilon
        for i in np.random.choice(range(n), self.outlier_num, replace=False):
            y[i] *= self.outlier
        return x, y
