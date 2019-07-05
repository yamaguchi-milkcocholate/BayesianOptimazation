import numpy as np
from scipy.stats import multivariate_normal


def example(x):
    x0, x1 = x[:, 0], x[:, 1]
    return -0.1 * (x0 ** 2 + x1 ** 2 - 16) ** 2 + 10 * np.sin(3 * x0)


class SchwefelsFunction:

    function_name = "Schwefel function"

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


class AlpineFunction:

    function_name = 'Alpine function'

    def __init__(self, dimensionality, dropout=None):
        def separable(xi): return np.abs(xi * np.sign(xi) + 0.1 * xi)
        self.separable = separable
        self.dimensionality = dimensionality

        if isinstance(dropout, list):
            self.active_dimension = [i for i in range(dimensionality) if i not in dropout]
        elif dropout is None:
            self.active_dimension = [i for i in range(dimensionality)]
        else:
            raise ValueError()

    def __call__(self, x):
        if x.shape[0] is 1 and x.shape[1] is self.dimensionality:
            x = x[0]
        elif x.shape[0] is self.dimensionality:
            x = x
        else:
            raise ValueError()
        return np.sum([self.separable(xi=x[i]) for i in range(len(x)) if i in self.active_dimension])

    def get_function_name(self):
        return self.function_name


class RosenbrockFunction:

    function_name = 'Rosenbrock function'

    def __init__(self, dimensionality):
        def sub_func(xi, xj): return 100 * (xj - xi * xi) * (xj - xi * xi) + (xi - 1) * (xi - 1)
        self.sub_func = sub_func
        self.dimensionality = dimensionality

    def __call__(self, x):
        if x.shape[0] is 1 and x.shape[1] is self.dimensionality:
            x = x[0]
        elif x.shape[0] is self.dimensionality:
            x = x
        else:
            raise ValueError()

        y = np.sum([self.sub_func(x[i], x[i + 1]) for i in range(len(x) - 1)])
        return y

    def get_function_name(self):
        return self.function_name
