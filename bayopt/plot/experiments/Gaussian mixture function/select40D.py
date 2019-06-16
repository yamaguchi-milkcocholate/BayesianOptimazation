from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='40D',
    method=['random_select', 'copy_select', 'mix_select', 'bo'], is_median=False)

plot_experiments(
    function_name='Gaussian mixture function', dim='40D',
    method=['random_select', 'copy_select', 'mix_select', 'bo'], is_median=True)
