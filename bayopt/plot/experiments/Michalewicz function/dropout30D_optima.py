from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Michalewicz function30-15', dim=['30D-15D', '30D', '30D_e0033t0500'],
    method=['copy', 'mix', 'bo', 'copy_select_acquisition_diff', 'mix_select_acquisition_diff'], is_median=True, iter_check=None, start=None, end=None, maximize=False,)
