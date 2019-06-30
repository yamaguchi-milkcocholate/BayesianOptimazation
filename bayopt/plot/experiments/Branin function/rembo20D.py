from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Brainin function', dim='20D',
    method=['REMBO_5'], is_median=False, iter_check=None, maxmize=False)

plot_experiments(
    function_name='Brainin function', dim='20D',
    method=['REMBO_5'], is_median=True, iter_check=None, maxmize=False)
