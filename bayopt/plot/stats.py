import numpy as np
import pandas as pd


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


def with_confidential(data):
    data = pd.DataFrame(data)

    if np.any(data.isnull().values):
        raise ValueError('Nan exists')

    mean = data.mean()
    std = data.std()

    data['mean'] = mean
    data['std'] = std

    print(data)

    if np.any(data.isnull().values):
        raise ValueError('Nan exists after inserting mean, std')

    return data
