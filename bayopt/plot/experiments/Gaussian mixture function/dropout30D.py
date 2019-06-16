from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='30D', method=['random', 'copy', 'mix', 'bo'], is_median=False)

plot_experiments(
    function_name='Gaussian mixture function', dim='30D', method=['random', 'copy', 'mix', 'bo'], is_median=True)
