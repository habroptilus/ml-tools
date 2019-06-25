import matplotlib.pyplot as plt
import numpy as np


class UnlabeledDataPlotter:

    def __init__(self, color="red"):
        self.color = color

    def plot(self, X, title, theta=None):
        """
        dim x = 2

        f(1,x1,x2) = theta[1] * x1 + theta[2] * x2 + theta[0]  
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(x1, x2, color=self.color, marker="x")
        if theta is not None:
            x = np.linspace(-6, 6, 100)
            y = -(theta[1] * x + theta[0]) / theta[2]
            ax.plot(x, y, color='black')

        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.show()
