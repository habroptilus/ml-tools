import matplotlib.pyplot as plt
import numpy as np


class LinearPlotter:
    """直線モデルに関してのPlotter"""

    def __init__(self, grid=100):
        self.grid = grid

    def plot_data(self, data_x, data_y, theta):
        """generatorが生成したdatasetの可視化.

        theta = (theta1, theta2) 
        y = theta1 * x + theta2で生成 
        """
        x = np.linspace(min(data_x), max(data_x), self.grid)
        y = x * theta[0] + theta[1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data_x, data_y)
        ax.plot(x, y, color='red')
        plt.title(
            f"Generated data (y={theta[0]:.2f}x + {theta[1]:.2f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def plot_model(self, data_x, data_y, theta):
        """modelの出力した回帰直線とdatasetの散布図を重ねて表示する."""
        x = np.linspace(min(data_x), max(data_x), self.grid)
        const = np.ones(self.grid)
        Phi = np.stack([const, x], axis=1)
        y = np.dot(Phi, theta)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data_x, data_y)
        ax.plot(x, y, color='red')
        plt.title("Regression result")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
