import numpy as np
from scipy.stats import multivariate_normal


def example(x):
    x0, x1 = x[:, 0], x[:, 1]
    return -0.1 * (x0 ** 2 + x1 ** 2 - 16) ** 2 + 10 * np.sin(3 * x0)


class GaussianMixtureFunction:

    def __init__(self, dim, mean_1, mean_2):
        self.dim = dim
        self.np_1 = multivariate_normal(np.full(dim, mean_1), np.diag(np.ones(dim)))
        self.np_2 = multivariate_normal(np.full(dim, mean_2), np.diag(np.ones(dim)))

    def __call__(self, x):
        return self.np_1.pdf(x) + 0.5 * self.np_2.pdf(x)
