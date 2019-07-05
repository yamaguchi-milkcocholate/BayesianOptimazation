from bayopt.plot.loader import load_experiments
from bayopt.plot.loader import load_experiments_theta
from bayopt.plot.loader import load_experiments_mask
from bayopt.plot.loader import load_experiments_model
from bayopt.plot.loader import load_experiments_evaluation
from bayopt.plot.staticplot import StaticPlot
from bayopt.plot.staticplot import BarPlot
from bayopt.plot.staticplot import HeatMap
from bayopt.plot.stats import maximum_locus
from bayopt.plot.stats import minimum_locus
from bayopt.plot.stats import with_confidential
from bayopt.plot.stats import histogram
from bayopt.plot.stats import pivot_table
from bayopt.plot.stats import to_zero_one
from bayopt.plot.stats import count_true
import numpy as np


def plot_experiments(function_name, dim, method, is_median=False, single=False, iter_check=None,
                     maximize=True, start=None, end=None, iteration=None, dirname=None, high=None, low=None, label=None):
def plot_experiments(function_name, dim, method, is_median=False, iter_check=None,
                     maximize=True, start=None, end=None, iteration=None, dirname=None,
                     high=None, low=None, label=None):

    if isinstance(dim, str):
        dim = [dim]

    data = dict()
    max_high = list()
    min_low = list()

    for fill in method:
        for d in dim:

            results = load_experiments(
                function_name=function_name,
                start=start,
                end=end,
                dim=d,
                feature=fill,
                iter_check=iter_check,
                dirname=dirname
            )

            if len(results) == 0:
                continue

            results_ = list()

            if maximize:
                results = results * -1

                for i in range(len(results)):
                    locus = maximum_locus(results[i])
                    if iteration:
                        locus = locus[:iteration]
                    results_.append(locus)
            else:
                for i in range(len(results)):
                    locus = minimum_locus(results[i])
                    if iteration:
                        locus = locus[:iteration]
                    results_.append(locus)

            results_ = np.array(results_)
            results_ = results_.T

            results_ = with_confidential(results_)

            x_axis = np.arange(0, len(results_))

            if is_median:
                prediction = results_['median']
                upper = results_['upper']
                lower = results_['lower']

            else:
                mean = results_['mean'].values
                std = results_['std'].values
                prediction = mean
                upper = mean + std
                lower = mean - std

            data[fill + d] = {'x_axis': x_axis, 'prediction': prediction, 'upper': upper, 'lower': lower}
            max_high.append(np.max(upper))
            min_low.append(np.min(lower))

    plot_all = StaticPlot()

    for key, data_ in data.items():

        if isinstance(label, dict):
            if key in label.keys():
                label_ = label[key]
            else:
                label_ = key
        else:
            label_ = key

        plot_all.add_data_set(x=data_['x_axis'], y=data_['prediction'], label=label_)
        plot_all.add_confidential_area(x=data_['x_axis'],
                                       upper_confidential_bound=data_['upper'], lower_confidential_bound=data_['lower'])

    if high is None:
        high = np.max(max_high)

    if low is None:
        low = np.min(min_low)

    plot_all.set_y(low_lim=low - 0.2, high_lim=high + 0.2)
    if is_median:
        plot_all.finish(option=function_name + '_' + '_median')
    else:
        plot_all.finish(option=function_name + '_' + '_mean')


def plot_experiment_evaluation(function_name, dim, method, created_at, update_check=None, maximize=True, high=None, low=None):
    evaluation = load_experiments_evaluation(
        function_name=function_name, dim=dim, feature=method, created_at=created_at, update_check=update_check
    )

    if maximize:
        evaluation = evaluation * -1

    x_axis = np.arange(0, len(evaluation))

    if high is None:
        high = np.max(evaluation)

    if low is None:
        low = np.min(evaluation)

    plot = StaticPlot()
    plot.add_data_set(x=x_axis, y=evaluation, label=method)
    plot.set_y(low_lim=low - 0.2, high_lim=high + 0.2)
    plot.finish(option=function_name + '_' + method)


def plot_experiment_theta(function_name, dim, method, created_at, update_check=None):
    theta = load_experiments_theta(
        function_name=function_name, dim=dim, feature=method, created_at=created_at, update_check=update_check
    )

    heat_map = HeatMap()

    heat_map.add_data_set(data=pivot_table(theta, value='theta', columns='iteration', index='dimension'), space=(0, 1))
    heat_map.finish(option=function_name + '_' + dim + '_theta')


def plot_experiment_theta_histogram(function_name, dim, method, created_at, update_idx=None, update_check=None):
    theta = load_experiments_theta(
        function_name=function_name, dim=dim, feature=method, created_at=created_at, update_check=update_check
    )

    if update_idx:
        theta = theta[update_idx]
    else:
        update_idx = len(theta) - 1
        theta = theta[-1]

    x, y = histogram(data=theta, start=0.0, stop=1.0, step=0.1)

    plot = BarPlot()
    plot.add_data_set(x=x, y=y)
    plot.finish(option=function_name + '_' + dim + '_theta_' + str(update_idx))


def plot_experiment_mask(function_name, dim, method, created_at, update_check=None):
    mask = load_experiments_mask(
        function_name=function_name, dim=dim, feature=method, created_at=created_at, update_check=update_check
    )

    mask = to_zero_one(mask)

    heat_map = HeatMap()

    heat_map.add_data_set(data=pivot_table(mask, value='theta', columns='iteration', index='dimension'), space=(0, 1))
    heat_map.finish(option=function_name + '_' + dim + '_mask')


def plot_experiment_subspace_dimensionality(function_name, dim, method, created_at, update_check=None):
    mask = load_experiments_mask(
        function_name=function_name, dim=dim, feature=method, created_at=created_at, update_check=update_check
    )

    iter_num = len(mask)

    # 1 step: ~100
    if len(mask) >= 100:
        y = mask[0:100]
        x = np.arange(0, 100)
    else:
        y = mask
        x = np.arange(0, len(mask))

    plot = BarPlot()
    plot.add_data_set(x=x, y=count_true(y), label='subspace dimensionality by 100')
    plot.finish(option=function_name + '_' + dim + 'subspace_dimensionality_by_100')

    # 5 step
    step = 5
    mask_ = count_true(mask)[np.arange(0, len(mask), step)]

    plot = BarPlot()
    x = np.arange(0, len(mask_))
    plot.add_data_set(x=x, y=mask_, label='subspace dimensionality')
    plot.set_x(x=x[np.arange(0, len(x), step)], x_ticks=np.arange(0, iter_num, step * step))
    plot.finish(option=function_name + '_' + dim + 'subspace_dimensionality')


def plot_experiment_model(function_name, dim, method, created_at, update_check=None):
    model = load_experiments_model(
        function_name=function_name, dim=dim, feature=method, created_at=created_at, update_check=update_check)

    model = model.drop('Iteration', axis=1)
    model = model.drop('GP_regression.Gaussian_noise.variance', axis=1)
    model = model.values

    heat_map = HeatMap()

    heat_map.add_data_set(data=pivot_table(
        model, value='value', columns='iteration', index='parameter'), space=(0, np.median(model)))
    heat_map.finish(option=function_name + '_' + dim + '_model')
