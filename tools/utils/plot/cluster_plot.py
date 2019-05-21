import matplotlib.pyplot as plt
import numpy as np


class ClusterPlotter:
    def __init__(self, colors=["r", "g", "b", "c", "m", "y", "b", "w"]):
        self.colors = colors

    def plot(self, X, y, title):
        """
        X data
        y label
        """
        plt.figure()
        for label in np.unique(y):
            x = X[y == label]
            plt.scatter(x[:, 0], x[:, 1], c=self.colors[label])
        plt.title(title)
        plt.show()
