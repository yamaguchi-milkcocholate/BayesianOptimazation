from bayopt.plot.utils import plot_experiments


plot_experiments(
    function_name='Michalewicz function60-30', dim=['60D', '60D-30D'],
    method=['bo', 'copy', 'mix'],
    is_median=False, iter_check=None, start=None, end=None, maximize=False,)
plot_experiments(
    function_name='Michalewicz function60-30', dim=['60D', '60D-30D'],
    method=['bo', 'copy', 'mix'],
    is_median=True, iter_check=None, start=None, end=None, maximize=False,)

