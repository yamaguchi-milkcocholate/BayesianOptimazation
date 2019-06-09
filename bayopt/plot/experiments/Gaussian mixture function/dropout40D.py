from bayopt.plot.loader import load_experiments
from bayopt.plot.staticplot import StaticPlot
from bayopt.plot.stats import maximum_locus
import numpy as np


for fill in ['random', 'copy', 'mix']:
    results = load_experiments(
        function_name='Gaussian mixture function',
        start=None,
        end=None,
        dim='40D',
        feature=fill,
        iter_check=501
    )

    results = results * -1
    results = maximum_locus(results)

    x_axis = np.arange(0, len(results))
    plot = StaticPlot()
    for n in range(results.shape[1]):
        plot.add_data_set(x=x_axis, y=results[:, n])

    plot.set_y(low_lim=0, high_lim=1)
    plot.finish(option='Gaussian mixture function_40D_' + fill)
