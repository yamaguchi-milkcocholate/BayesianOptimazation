from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim=['30D-25D', '30D_e0033t0500'],
    method=['copy', 'mix', 'copy_select_objective', 'mix_select_objective'], is_median=False, maximize=True,
    iter_check=None, start=None, end=None, iteration=500, dirname='dropout30D-25D_obj',
    label={'copy_select_objective30D_e0033t0500': 'select_copy_30D',
           'mix_select_objective30D_e0033t0500': 'select_mix_30D'}
)

plot_experiments(
    function_name='Gaussian mixture function', dim=['30D-25D', '30D_e0033t0500'],
    method=['copy', 'mix', 'copy_select_objective', 'mix_select_objective'], is_median=True, maximize=True,
    iter_check=None, start=None, end=None, iteration=500, dirname='dropout30D-25D_obj',
    label={'copy_select_objective30D_e0033t0500': 'select_copy_30D',
           'mix_select_objective30D_e0033t0500': 'select_mix_30D'}
)
