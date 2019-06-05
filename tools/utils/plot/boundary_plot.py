import matplotlib.pyplot as plt
import numpy as np


class BinaryDataPlotter:
    """binary dataをplotする.分類境界も表示する."""

    def __init__(self, pos_color="blue", neg_color="red"):
        self.pos_color = pos_color
        self.neg_color = neg_color

    def plot(self, X, y, title, theta=None):
        """
        y={1,-1}, 
        dim x = 2

        f(1,x1,x2) = theta1 * x1 + theta2 * x2 + theta3  
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(x1[y == 1], x2[y == 1], color=self.pos_color, marker="o")
        ax.scatter(x1[y == -1], x2[y == -1], color=self.neg_color, marker="x")
        if theta is not None:
            x = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
            y = -(theta[1] * x + theta[0]) / theta[2]
            ax.plot(x, y, color='black')

        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
