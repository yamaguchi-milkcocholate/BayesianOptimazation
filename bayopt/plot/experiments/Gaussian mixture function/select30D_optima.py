from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Gaussian mixture function', dim='30D_e0183t0167',
    method=['random_select_objective', 'copy_select_objective', 'mix_select_objective'], maximize=True, is_median=False, start=None, end=None)

plot_experiments(
    function_name='Gaussian mixture function', dim='30D_e0183t0167',
    method=['random_select_objective', 'copy_select_objective', 'mix_select_objective'], maximize=True, is_median=True, start=None, end=None)
