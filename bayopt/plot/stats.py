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
        raise ValueError('shape (m, 0)')

    data_ = list()
    minimum = 0

    for i in range(len(data)):
        if minimum > data[i]:
            minimum = data[i]

        data_.append(minimum)

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


def histogram(data, start, stop, step):
    x = np.arange(start=start, stop=stop + step, step=step)
    y = list()

    for i in range(len(x) - 1):
        if i == (len(x) - 2):
            y.append(np.count_nonzero((x[i] <= data) & (data <= x[i + 1])))
        else:
            y.append(np.count_nonzero((x[i] <= data) & (data < x[i + 1])))

    return np.round(x[:-1], 1).astype(np.str), np.array(y)


def pivot_table(data, value, columns, index):

    df = {value: list(), columns: list(), index: list()}

    for iteration_idx in range(len(data)):
        for dim_idx in range(len(data[iteration_idx])):

            df[value].append(data[iteration_idx][dim_idx])
            df[columns].append(iteration_idx + 1)
            df[index].append(dim_idx + 1)

    df = pd.DataFrame(df)
    df_pivot = pd.pivot_table(data=df, values=value, columns=columns, index=index, aggfunc=np.mean)
    return df_pivot


def to_zero_one(data):
    def f(x):
        if x == 'True':
            return 1
        else:
            return 0

    return np.vectorize(f)(data)


def count_true(data):
    data = to_zero_one(data)
    return np.apply_along_axis(np.sum, 1, data)
