import numpy as np


def example(x):
    x0, x1 = x[:, 0], x[:, 1]
    return -0.1 * (x0 ** 2 + x1 ** 2 - 16) ** 2 + 10 * np.sin(3 * x0)