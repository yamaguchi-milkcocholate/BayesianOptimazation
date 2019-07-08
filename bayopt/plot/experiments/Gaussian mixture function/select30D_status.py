from bayopt.plot.utils import plot_experiments_theta
from bayopt.plot.utils import plot_experiments_mask


plot_experiments_theta(
        function_name='Gaussian mixture function', dim='30D_e0033t0500',
        method='mix_select_acquisition_diff', dirname='dropout30D-5D_acqdiff'
)


plot_experiments_mask(
        function_name='Gaussian mixture function', dim='30D_e0033t0500',
        method='mix_select_acquisition_diff', dirname='dropout30D-5D_acqdiff'
)
