from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim=['30D-5D', '30D'],
    method=['random', 'copy', 'mix', 'bo'], is_median=False, maximize=True, iter_check=None,
    start=None, end=None, iteration=500)

plot_experiments(
    function_name='Gaussian mixture function', dim=['30D-5D', '30D'],
    method=['random', 'copy', 'mix', 'bo'], is_median=True, maximize=True, iter_check=None,
    start=None, end=None, iteration=500)
