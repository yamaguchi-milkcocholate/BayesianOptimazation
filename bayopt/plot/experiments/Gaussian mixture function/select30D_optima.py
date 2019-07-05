from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim=['30D_e0033t0167', '30D-15D', '30D_e0033t0500'],
    method=['copy_select_acquisition', 'copy_select_objective', 'mix_select_objective',
            'mix_select_acquisition', 'copy', 'mix'],
    maximize=True, is_median=False, start=None, end=None)

plot_experiments(
    function_name='Gaussian mixture function', dim=['30D_e0033t0167', '30D-15D', '30D_e0033t0500'],
    method=['copy_select_acquisition', 'copy_select_objective', 'mix_select_objective',
            'mix_select_acquisition', 'copy', 'mix'],
    maximize=True, is_median=True, start=None, end=None)