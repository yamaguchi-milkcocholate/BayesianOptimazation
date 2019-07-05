from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim=['30D-25D', '30D_e0033t0500'],
    method=['copy', 'mix', 'copy_select_acquisition_diff', 'mix_select_acquisition_diff'], is_median=False, maximize=True, iter_check=None,
    start=None, end=None, iteration=500, dirname='dropout30D-25D')

plot_experiments(
    function_name='Gaussian mixture function', dim=['30D-25D', '30D_e0033t0500'],
    method=['copy', 'mix', 'copy_select_acquisition_diff', 'mix_select_acquisition_diff'], is_median=True, maximize=True, iter_check=None,
    start=None, end=None, iteration=500, dirname='dropout30D-25D')
