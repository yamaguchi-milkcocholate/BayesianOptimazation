from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='40D',
    method=['random', 'copy', 'mix', 'bo'], is_median=False, maximize=True, iter_check=500, start=None, end=None)

plot_experiments(
    function_name='Gaussian mixture function', dim='40D',
    method=['random', 'copy', 'mix', 'bo'], is_median=True, maximize=True, iter_check=500, start=None, end=None)
