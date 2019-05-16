import matplotlib.pyplot as plt
import numpy as np


class ClusterPlotter:
    def __init__(self, colors=["r", "g", "b", "c", "m", "y", "b", "w"]):
        self.colors = colors

    def plot(self, X, y):
        """
        X data
        y label
        """
        for label in np.unique(y):
            plt.scatter(X[y == label], c=colors[label])
        plt.show()
