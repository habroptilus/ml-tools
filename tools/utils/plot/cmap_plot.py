import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot3D(data, c, title):
    """3次元空間にcに関してのグラデーションつきの散布図を描画."""
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker="o",
               linestyle='None', c=c, cmap='rainbow')
    plt.title(title)
    plt.show()


def plot2D(data, c, title):
    """2次元空間にcに関してのグラデーションつきの散布図を描画."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.scatter(data[:, 0], data[:, 1], marker="o",
               linestyle='None', c=c, cmap='rainbow')
    plt.title(title)
    plt.show()
