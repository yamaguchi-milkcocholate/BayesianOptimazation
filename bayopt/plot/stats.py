import numpy as np


def maximum_locus(data):
    if len(data.shape) is not 1:
        raise ValueError('shape (n, 0)')

    data_ = list()
    maximum = 0

    for i in range(len(data)):
        if maximum < data[i]:
            maximum = data[i]

        data_.append(maximum)

    return np.array(data_)


def minimum_locus(data):
    if len(data.shape) is not 1:
        raise ValueError('shape (n, 0)')

    data_ = list()
    minimum = np.inf

    for i in range(len(data)):
        if data[i] < minimum:
            minimum = data[i]

        data_.append(minimum)

    return np.array(data_)
