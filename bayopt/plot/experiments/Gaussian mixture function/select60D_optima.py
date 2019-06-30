from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='60D',
    method=['random_select', 'copy_select', 'mix_select'], maximize=True, is_median=False, iter_check=500,
    start=None, end=None)

plot_experiments(
    function_name='Gaussian mixture function', dim='60D',
    method=['random_select', 'copy_select', 'mix_select'], maximize=True, is_median=True, iter_check=500,
    start=None, end=None)
