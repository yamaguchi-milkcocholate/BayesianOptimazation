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

    if len(data) == 0:
        raise ValueError('No Data')

    if np.any(data.isnull().values):
        print(data)
        raise ValueError('Nan exists')

    mean = data.mean(axis='columns')

    if len(data.columns) is not 1:
        std = data.std(axis='columns')
    else:
        std = np.zeros(len(data))

    median = data.median(axis='columns')
    quantile = data.quantile(q=[0.25, 0.75], axis=1)

    data['mean'] = mean
    data['std'] = std
    data['median'] = median
    data['upper'] = quantile.loc[0.75, :]
    data['lower'] = quantile.loc[0.25, :]

    if np.any(data.isnull().values):
        print(data)
        raise ValueError('Nan exists after inserting mean, std')

    return data
