from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Michalewicz function10-10', dim=['10D-5D'],
    method=['copy', 'mix'], is_median=False, iter_check=None, start=None, end=None, maximize=False,)
plot_experiments(
    function_name='Michalewicz function10-10', dim=['10D-5D'],
    method=['copy', 'mix'], is_median=True, iter_check=None, start=None, end=None, maximize=False,)
