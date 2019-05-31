import numpy as np
from scipy.stats import multivariate_normal


def example(x):
    x0, x1 = x[:, 0], x[:, 1]
    return -0.1 * (x0 ** 2 + x1 ** 2 - 16) ** 2 + 10 * np.sin(3 * x0)


def gauss(x, mu, sigma):
    return multivariate_normal.pdf(x, mean=mu, cov=sigma)


def gaussian_mixture_function(x):
    np_1 = multivariate_normal.pdf(x, mean=np.full(len(x), 2), cov=np.diag(np.ones(len(x))))
    np_2 = multivariate_normal.pdf(x, mean=np.full(len(x), 3), cov=np.diag(np.ones(len(x))))
    return np_1 + np_2 * 0.5


class GaussianMixtureFunction:

    def __init__(self, dim):
        self.dim = dim
        self.np_1 = multivariate_normal(np.full(dim, 2), np.diag(np.ones(dim)))
        self.np_2 = multivariate_normal(np.full(dim, 3), np.diag(np.ones(dim)))

    def __call__(self, x):
        return self.np_1.pdf(x) + 0.5 * self.np_2.pdf(x)
