from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim=['60D_e0017t0500', '60D-30D'],
    method=['copy_select_acquisition_diff', 'copy'], maximize=True, is_median=False, iter_check=None,
    start=None, end=None)

plot_experiments(
    function_name='Gaussian mixture function', dim=['60D_e0017t0500', '60D-30D'],
    method=['copy_select_acquisition_diff', 'copy'], maximize=True, is_median=True, iter_check=None,
    start=None, end=None)
