from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim=['5D_e0200t0400'],
    method=['copy_select_objective'],
    maximize=True, is_median=False, start=None, end=None)

plot_experiments(
    function_name='Gaussian mixture function', dim=['5D_e0200t0400'],
    method=['copy_select_objective'],
    maximize=True, is_median=True, start=None, end=None)
