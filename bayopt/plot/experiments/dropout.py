from bayopt.plot.loader import load_experiments
from bayopt.plot.staticplot import StaticPlot
import numpy as np

results = load_experiments(
    function_name='Gaussian mixture function',
    start=None,
    end=None,
    dim='5D',
    feature='bo'
)

results = results * -1

x_axis = np.arange(0, len(results))
plot = StaticPlot()
for n in range(results.shape[1]):
    plot.add_data_set(x=x_axis, y=results[:, n])

plot.finish()
