from bayopt.plot.utils import plot_experiments_theta
from bayopt.plot.utils import plot_experiments_mask


plot_experiments_theta(
        function_name='Gaussian mixture function', dim='30D_e0033t0500',
        method='copy_select_acquisition', dirname='dropout30D-5D_acq'
)


plot_experiments_mask(
        function_name='Gaussian mixture function', dim='30D_e0033t0500',
        method='copy_select_acquisition', dirname='dropout30D-5D_acq'
)
