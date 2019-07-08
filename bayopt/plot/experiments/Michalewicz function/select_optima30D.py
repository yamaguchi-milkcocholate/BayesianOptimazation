from bayopt.plot.utils import plot_experiments


plot_experiments(
    function_name='Michalewicz function30-15', dim=['30D_e0033t0500', '30D-15D', '30D'],
    method=['copy_select_acquisition_diff', 'mix_select_acquisition_diff', 'copy', 'mix', 'bo'],
    is_median=False, iter_check=None, start=None, end=None, maximize=False,)
plot_experiments(
    function_name='Michalewicz function30-15', dim=['30D_e0033t0500', '30D-15D', '30D'],
    method=['copy_select_acquisition_diff', 'mix_select_acquisition_diff', 'copy', 'mix', 'bo'],
    is_median=True, iter_check=None, start=None, end=None, maximize=False,)

