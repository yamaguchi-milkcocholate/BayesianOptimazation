from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='30D',
    method=['copy_select', ], is_median=False)

plot_experiments(
    function_name='Gaussian mixture function', dim='30D',
    method=['random_select', 'copy_select', 'mix_select'], is_median=True)
