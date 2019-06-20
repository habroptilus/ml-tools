import matplotlib.pyplot as plt
import numpy as np


class BinaryLabeledPlotter:
    """binary dataをplotする.分類境界も表示(境界が直線のとき可能)."""

    def __init__(self, pos_color="blue", neg_color="red", line_styles=["solid", "dashed"], line_colors=["green", "black"]):
        self.pos_color = pos_color
        self.neg_color = neg_color
        self.line_styles = line_styles
        self.line_colors = line_colors

    def plot(self, X, y, title, models=None):
        """
        y={1,-1}, 
        dim x = 2

        f(1,x1,x2) = theta[1] * x1 + theta[2] * x2 + theta[0]  
        """
        x1 = X[:, 0]
        x2 = X[:, 1]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x2_max = max(x2) + 10
        x2_min = min(x2) - 10

        ax.scatter(x1[y == 1], x2[y == 1], color=self.pos_color,
                   marker="o", label="pos")
        ax.scatter(x1[y == -1], x2[y == -1],
                   color=self.neg_color, marker="x", label="neg")
        if models is not None:
            for i, d in enumerate(models):
                theta = d["theta"]
                model = d["model"]
                line_x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 10000)
                line_x2 = -(theta[1] * line_x1 + theta[0]) / theta[2]
                line_x1 = line_x1[(x2_min < line_x2) & (line_x2 < x2_max)]
                line_x2 = line_x2[(x2_min < line_x2) & (line_x2 < x2_max)]
                ax.plot(
                    line_x1, line_x2, color=self.line_colors[i], linestyle=self.line_styles[i], label=model)

        plt.title(title)
        plt.legend()
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
