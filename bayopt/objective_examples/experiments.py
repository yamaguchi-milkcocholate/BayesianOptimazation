import numpy as np
from scipy.stats import multivariate_normal


def example(x):
    x0, x1 = x[:, 0], x[:, 1]
    return -0.1 * (x0 ** 2 + x1 ** 2 - 16) ** 2 + 10 * np.sin(3 * x0)


class SchwefelsFunction:

    function_name = "Schwefel's function"

    def __call__(self, x):
        return np.sum([(np.sum([x[dj] for dj in range(di + 1)]) ** 2) for di in range(len(x))])

    def get_function_name(self):
        return self.function_name


class GaussianMixtureFunction:

    function_name = 'Gaussian mixture function'

    def __init__(self, dim, mean_1, mean_2):
        self.dim = dim
        self.np_1 = multivariate_normal(np.full(dim, mean_1), np.diag(np.ones(dim)))
        self.np_2 = multivariate_normal(np.full(dim, mean_2), np.diag(np.ones(dim)))
        self.max_value = self.np_1.pdf(np.full(dim, mean_1)) + 0.5 * self.np_2.pdf(np.full(dim, mean_1))

    def __call__(self, x):
        return (self.np_1.pdf(x) + 0.5 * self.np_2.pdf(x)) / self.max_value

    def get_function_name(self):
        return self.function_name


class BraininFunction:

    function_name = 'Brainin function'

    def __init__(self, effective1, effective2):
        self.effective1 = effective1
        self.effective2 = effective2

    def __call__(self, x):
        x1 = x[0][self.effective1]
        x2 = x[0][self.effective2]
        return ((x2 - (5.1*(x2**2)/(4*(np.pi**2))) + 5*x1/np.pi - 6)**2 + 10*(1-1/(8*np.pi))*np.cos(x1) + 10) - 0.397887

    def get_function_name(self):
        return self.function_name
