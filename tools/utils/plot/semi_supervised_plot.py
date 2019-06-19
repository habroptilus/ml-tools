import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


class SemiSupervisedPlotter:
    def __init__(self):
        pass

    def plot(self, X, y, title, model=None, resolution=10):
        """
        X data (dim = 2)
        y label {+1, None(unlabeled), -1}
        """
        plt.figure()
        if model is not None:
            cmap = ListedColormap(('magenta', 'cyan'))
            x1_min, x1_max = X[:, 0].min() - 50, X[:, 0].max() + 50
            x2_min, x2_max = X[:, 1].min() - 50, X[:, 1].max() + 50
            x1_mesh, x2_mesh = np.meshgrid(
                np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
            Z = model.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
            Z = Z.reshape(x1_mesh.shape)
            plt.contourf(x1_mesh, x2_mesh, Z, alpha=0.4, cmap=cmap)

        x = X[y == 1]
        plt.scatter(x[:, 0], x[:, 1], c="blue", marker="o", label="pos")
        x = X[y == -1]
        plt.scatter(x[:, 0], x[:, 1], c="red", marker="x", label="neg")
        x = X[y == None]
        plt.scatter(x[:, 0], x[:, 1], c="black", marker=".", label="unlabeled")
        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.show()
