from bayopt.plot.utils import plot_experiments

plot_experiments(
    function_name='Michalewicz function10-10', dim=['10D-5D', '10D-2D', '10D'],
    method=['copy', 'mix', 'bo', 'copy_select_objective', 'mix_select_objective'], is_median=False, iter_check=None,
    start=None, end=None, maximize=False,
)
plot_experiments(
    function_name='Michalewicz function10-10', dim=['10D-5D', '10D-2D', '10D', '10D_e0100t0500'],
    method=['copy', 'mix', 'bo', 'copy_select_objective', 'mix_select_objective'], is_median=True, iter_check=None,
    start=None, end=None, maximize=False,
    label={'copy_select_objective10D_e0100t0500': 'teian_copy', 'mix_select_objective10D_e0100t0500': 'teian_mix'}
)
