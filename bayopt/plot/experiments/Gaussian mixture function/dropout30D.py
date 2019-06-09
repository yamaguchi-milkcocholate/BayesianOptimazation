from bayopt.plot.loader import load_experiments
from bayopt.plot.staticplot import StaticPlot
from bayopt.plot.stats import maximum_locus
from bayopt.plot.stats import with_confidential
import numpy as np


for fill in ['random', 'copy', 'mix']:
    results = load_experiments(
        function_name='Gaussian mixture function',
        start=None,
        end=None,
        dim='30D',
        feature=fill,
        iter_check=501
    )

    data = list()

    for i in range(len(results)):
        data.append(maximum_locus(results[i]))

    data = np.array(data)
    data = data * -1
    data = data.T

    results_ = with_confidential(data)

    mean = results_['mean'].values
    std = results_['std'].values

    x_axis = np.arange(0, len(results_))
    plot = StaticPlot()

    plot.add_data_set(x=x_axis, y=mean)
    plot.add_confidential_area(x=x_axis, mean=mean, std=std)

    plot.set_y(low_lim=0, high_lim=1)
    plot.finish(option='Gaussian mixture function_30D_' + fill)
