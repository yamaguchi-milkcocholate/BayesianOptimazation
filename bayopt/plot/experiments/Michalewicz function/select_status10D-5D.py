from bayopt.plot.utils import plot_experiments_theta
from bayopt.plot.utils import plot_experiments_mask


plot_experiments_theta(
        function_name='Michalewicz function10-5', dim='10D_e0100t0500',
        method='copy_select_acquisition', dirname=None
)


plot_experiments_mask(
        function_name='Michalewicz function10-5', dim='10D_e0100t0500',
        method='copy_select_acquisition', dirname=None
)
