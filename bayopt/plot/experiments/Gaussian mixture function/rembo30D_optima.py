from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='30D',
    method=['REMBO_5'], is_median=False, iter_check=500, maximize=True, start=None, end=None)

plot_experiments(
    function_name='Gaussian mixture function', dim='30D',
    method=['REMBO_5'], is_median=True, iter_check=500, maximize=True, start=None, end=None)
