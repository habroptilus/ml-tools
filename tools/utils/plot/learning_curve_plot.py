import matplotlib.pyplot as plt
import numpy as np


class LearningCurvePlotter:
    def __init__(self):
        pass

    def plot(self, losses):
        plt.figure()
        plt.title("Learning Curve")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        x = np.arange(len(losses))
        plt.plot(x, losses, color="r", label="Training loss")
