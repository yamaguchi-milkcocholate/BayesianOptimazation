from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Rosenbrock function', dim='5D',
    method=['bo'], is_median=False, iter_check=None, start=None, end=None, maximize=False, high=50)
plot_experiments(
    function_name='Rosenbrock function', dim='5D',
    method=['bo'], is_median=True, iter_check=None, start=None, end=None, maximize=False, high=50)
