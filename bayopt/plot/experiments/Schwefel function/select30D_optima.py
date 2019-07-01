from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name="Schwefel function", dim='30D_e0183t0167',
    method=['random_select_acquisition', 'copy_select_acquisition', 'random_select_objective', 'copy_select_objective'],
    maximize=False, is_median=False, start=None, end=None)

plot_experiments(
    function_name="Schwefel function", dim='30D_e0183t0167',
    method=['random_select_acquisition', 'copy_select_acquisition', 'random_select_objective', 'copy_select_objective'],
    maximize=False, is_median=True, start=None, end=None)
