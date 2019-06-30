from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='30D-15D',
    method=['random', 'copy', 'mix'], is_median=False, maximize=True, iter_check=500, start=None, end=None)

plot_experiments(
    function_name='Gaussian mixture function', dim='30D-15D',
    method=['random', 'copy', 'mix'], is_median=True, maximize=True, iter_check=500, start=None, end=None)
