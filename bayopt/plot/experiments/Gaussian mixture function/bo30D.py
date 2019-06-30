from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='30D',
    method=['bo'], is_median=False, iter_check=500, start='2019-06-28 09:21:21', end='2019-06-28 09:21:21')
plot_experiments(
    function_name='Gaussian mixture function', dim='30D',
    method=['bo'], is_median=True, iter_check=500, start='2019-06-28 09:21:21', end='2019-06-28 09:21:21')
