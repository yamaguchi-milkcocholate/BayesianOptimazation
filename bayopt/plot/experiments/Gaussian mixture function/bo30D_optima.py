from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='30D',
    method=['bo'], is_median=False, iter_check=None, start=None, end=None)
plot_experiments(
    function_name='Gaussian mixture function', dim='30D',
    method=['bo'], is_median=True, iter_check=None, start=None, end=None)
