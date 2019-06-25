from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='50D',
    method=['random_select', 'copy_select', 'mix_select'], is_median=False, iter_check=500)

plot_experiments(
    function_name='Gaussian mixture function', dim='50D',
    method=['random_select', 'copy_select', 'mix_select'], is_median=True, iter_check=500)
