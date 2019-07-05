from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Michalewicz function', dim=['30D-15D', '30D-25D'],
    method=['random', 'copy', 'mix'], is_median=False, iter_check=None, start=None, end=None, maximize=False,)
plot_experiments(
    function_name='Michalewicz function', dim=['30D-15D', '30D-25D'],
    method=['random', 'copy', 'mix'], is_median=True, iter_check=None, start=None, end=None, maximize=False,)
