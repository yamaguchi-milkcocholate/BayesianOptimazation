from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Brainin function', dim='20D',
    method=['bo'], is_median=False, iter_check=None,
    start='2019-06-28 12:28:25', end='2019-06-28 12:28:25', maximize=False)
plot_experiments(
    function_name='Brainin function', dim='20D',
    method=['bo'], is_median=True, iter_check=None,
    start='2019-06-28 12:28:25', end='2019-06-28 12:28:25', maximize=False)
