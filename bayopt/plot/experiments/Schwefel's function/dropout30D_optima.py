from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Schwefel function', dim='30D-15D',
    method=['random', 'copy', 'mix'], is_median=False, maximize=False, iter_check=250, start=None, end=None)

plot_experiments(
    function_name='Schwefel function', dim='30D-15D',
    method=['random', 'copy', 'mix'], is_median=True, maximize=False, iter_check=250, start=None, end=None)
