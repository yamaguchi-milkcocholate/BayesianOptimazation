from bayopt import definitions
from bayopt.clock.clock import from_str
import os
import csv
import numpy as np


def load_experiments(function_name, dim, feature, start=None, end=None):
    experiments = load_files(
        function_name=function_name, start=start, end=end, dim=dim, feature=feature)
    results = list()
    for expt in experiments:
        evaluation_file = expt + '/evaluation.csv'
        y = csv_to_numpy(file=evaluation_file)
        results.append(y)

    return np.array(results, dtype=np.float).T


def csv_to_numpy(file):
    y = list()

    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # ヘッダーを読み飛ばしたい時

        for row in reader:
            y.append(row[1])
    return np.array(y)


def load_files(function_name, start=None, end=None, **kwargs):
    """
    :param function_name: string
    :param start: string
    :param end:  string
    :return: list
    """
    storage_dir = definitions.ROOT_DIR + '/storage/' + function_name
    experiments = os.listdir(storage_dir)

    masked = list()

    if start:
        start = from_str(start)
    if end:
        end = from_str(end)
    for expt in experiments:
        dt, tm, dim, feature = expt.split(' ')
        expt_time = from_str(dt + ' ' + tm)

        is_append = True

        if start:
            if expt_time <= start:
                is_append = False

        if end:
            if end <= expt_time:
                is_append = False

        for kwd in kwargs:
            if not (kwargs[kwd] in dim or kwargs[kwd] in feature):
                is_append = False

        if is_append:
            masked.append(storage_dir + '/' + expt)

    return masked
