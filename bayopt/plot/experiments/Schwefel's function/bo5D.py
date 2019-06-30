from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name="Schwefel's function", dim='5D',
    method=['bo'], is_median=False, maximize=False, iter_check=None, start=None, end=None)

plot_experiments(
    function_name="Schwefel's function", dim='5D',
    method=['bo'], is_median=True, maximize=False, iter_check=None, start=None, end=None)
