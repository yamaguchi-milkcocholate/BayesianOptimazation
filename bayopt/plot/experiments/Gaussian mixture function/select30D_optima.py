from bayopt.plot.utils import plot_experiments


plot_experiments(
    function_name='Gaussian mixture function', dim=['30D_e0033t0500', '30D-25D', '30D'],
    method=['copy', 'mix', 'bo', 'copy_select_objective', 'mix_select_objective'],
    maximize=True, is_median=False, start=None, end=None, dirname='dropout30D-25D_obj')

plot_experiments(
    function_name='Gaussian mixture function', dim=['30D_e0033t0500', '30D-25D', '30D'],
    method=['copy', 'mix', 'bo', 'copy_select_objective', 'mix_select_objective'],
    maximize=True, is_median=True, start=None, end=None,  dirname='dropout30D-25D_obj',
    label={'copy_select_objective30D_e0033t0500': 'select_copy_30D',
           'mix_select_objective30D_e0033t0500': 'select_mix_30D'}
)
