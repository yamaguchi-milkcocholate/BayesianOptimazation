from bayopt.plot.loader import load_experiments
from bayopt.plot.staticplot import StaticPlot
from bayopt.plot.stats import maximum_locus
from bayopt.plot.stats import with_confidential
import numpy as np

all_ = dict()

for fill in ['random_select', 'copy_select', 'mix_select']:
    results = load_experiments(
        function_name='Gaussian mixture function',
        start=None,
        end=None,
        dim='30D',
        feature=fill,
    )
    results = results * -1

    data = list()

    for i in range(len(results)):
        data.append(maximum_locus(results[i]))

    data = np.array(data)
    data = data.T

    results_ = with_confidential(data)

    mean = results_['mean'].values
    std = results_['std'].values

    x_axis = np.arange(0, len(results_))
    plot = StaticPlot()

    plot.add_data_set(x=x_axis, y=mean)
    plot.add_confidential_area(x=x_axis, mean=mean, std=std)

    all_[fill] = {'x_axis': x_axis, 'mean': mean, 'std': std}

    plot.set_y(low_lim=-0.2, high_lim=1.2)
    plot.finish(option='Gaussian mixture function_30D_' + fill)

plot_all = StaticPlot()

for key, data_ in all_.items():
    plot_all.add_data_set(x=data_['x_axis'], y=data_['mean'], label=key)
    plot_all.add_confidential_area(x=data_['x_axis'], mean=data_['mean'], std=data_['std'])


plot_all.set_y(low_lim=-0.2, high_lim=1.2)
plot_all.finish(option='Gaussian mixture function_30D')
