import numpy as np


class ClusteredDataGenerator:
    def __init__(self, mus, sigmas):
        if mus.shape[0] != sigmas.shape[0]:
            raise Exception("The numbers of class are different.")
        if mus.shape[1] != sigmas.shape[1]:
            raise Exception("The numbers of dim are different.")
        if sigmas.shape[1] != sigmas.shape[2]:
            raise Exception("Sigma should be square matrix.")
        self.class_num = sigmas.shape[0]
        self.mus = mus
        self.sigmas = sigmas

    def run(self, N):
        """
        :params N: the number of samples in each class
        """
        X = []
        Y = []
        for i in range(self.class_num):
            x = np.random.multivariate_normal(self.mus[i], self.sigmas[i], N)
            y = [i] * N
            X.extend(x)
            Y.extend(y)

        return np.array(X), np.array(Y)
