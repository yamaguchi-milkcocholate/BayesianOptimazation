from bayopt import definitions
from bayopt.clock.clock import from_str
from bayopt.utils.utils import rmdir_when_any
import pandas as pd
import os
import csv
import numpy as np


def load_experiments(function_name, dim, feature, start=None, end=None, iter_check=None, dirname=None):
    experiments = load_files(
        function_name=function_name, start=start, end=end, dim=dim, feature=feature, dirname=dirname)

    results = list()
    for expt in experiments:
        evaluation_file = expt + '/evaluation.csv'
        y = csv_to_numpy(file=evaluation_file)
        y = y[:, 1]

        if iter_check:
            if len(y) < iter_check:
                print('Error in ' + expt + ': expect ' + str(iter_check) + ' given ' + str(len(y)))
                # rmdir_when_any(expt)
                raise ValueError('iterations is not enough')

        results.append(y)

    results = make_uniform_by_length(results)

    return np.array(results, dtype=np.float)


def load_experiments_evaluation(function_name, dim, feature, created_at, update_check=None):
    expt = _load_experiment(function_name=function_name, created_at=created_at, dim=dim,
                            feature=feature)

    expt_file = expt + '/evaluation.csv'
    data = csv_to_numpy(expt_file, header=True)
    data = data[:, 1]

    if update_check:
        if len(data) < update_check:
            print('expect ' + str(update_check) + ' given ' + str(len(data)))

            raise ValueError('Not Enough')

    return data


def load_experiments_theta(function_name, dim, feature, created_at, update_check=None):
    expt = _load_experiment(function_name=function_name, created_at=created_at, dim=dim,
                            feature=feature)

    expt_file = expt + '/distribution.csv'
    data = csv_to_numpy(expt_file, header=False)

    if update_check:
        if len(data) < update_check:
            print('expect ' + str(update_check) + ' given ' + str(len(data)))

            raise ValueError('Not Enough')

    return data


def load_experiments_mask(function_name, dim, feature, created_at, update_check=None):
    expt = _load_experiment(function_name=function_name, created_at=created_at, dim=dim,
                            feature=feature)

    expt_file = expt + '/mask.csv'
    data = csv_to_numpy(expt_file, header=False, dtype='str')

    if update_check:
        if len(data) < update_check:
            print('expect ' + str(update_check) + ' given ' + str(len(data)))

            raise ValueError('Not Enough')

    return data


def load_experiments_model(function_name, dim, feature, created_at, update_check=None):
    expt = _load_experiment(function_name=function_name, created_at=created_at, dim=dim,
                            feature=feature)

    expt_file = expt + '/model.csv'

    data = pd.read_csv(expt_file, delimiter='\t', dtype=float)

    if update_check:
        if len(data) < update_check:
            print('expect ' + str(update_check) + ' given ' + str(len(data)))

            raise ValueError('Not Enough')

    return data


def _load_experiment(function_name, created_at, dim, feature):
    experiments = load_files(
        function_name=function_name, start=created_at, end=created_at, dim=dim, feature=feature)

    if len(experiments) == 0:
        raise FileNotFoundError('zero experiments')

    if len(experiments) > 1:
        raise ValueError('2 more file exist.')

    expt = experiments[0]

    print(expt)

    return expt


def csv_to_numpy(file, header=True, dtype='float'):
    y = list()

    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        if header:
            next(reader)

        for row in reader:
            y.append(row)

    if dtype is 'float':
        return np.array(y, dtype=np.float)

    elif dtype is 'str':
        return np.array(y, dtype=np.str)


def load_files(function_name, start=None, end=None, dirname=None, **kwargs):
    print(dirname)
    if dirname:
        storage_dir = definitions.ROOT_DIR + '/storage/' + function_name + '/' + dirname
    else:
        storage_dir = definitions.ROOT_DIR + '/storage/' + function_name

    print(storage_dir)
    experiments = os.listdir(storage_dir)

    masked = list()

    if start:
        start = from_str(start)
    if end:
        end = from_str(end)
    for expt in experiments:
        try:
            dt, tm, dim, feature = expt.split(' ')
        except ValueError as e:
            print('discard: ' + expt)
            continue

        expt_time = from_str(dt + ' ' + tm)

        is_append = True

        if start:
            if expt_time < start:
                is_append = False

        if end:
            if end < expt_time:
                is_append = False

        for kwd in kwargs:
            if not (kwargs[kwd] == dim or kwargs[kwd] == feature):
                is_append = False

        if is_append:
            masked.append(storage_dir + '/' + expt)

    return masked


def make_uniform_by_length(list_obj):
    if len(list_obj) is 0:
        return list()

    list_ = list()

    lengths = [len(el) for el in list_obj]

    min_len = min(lengths)

    for el in list_obj:
        list_.append(el[0:min_len])

    return list_
