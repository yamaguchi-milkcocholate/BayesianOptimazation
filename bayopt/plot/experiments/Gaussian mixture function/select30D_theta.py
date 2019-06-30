from bayopt.plot.utils import plot_experiment_theta_histogram
from bayopt.plot.utils import plot_experiment_theta
from bayopt.plot.utils import plot_experiment_mask
from bayopt.plot.utils import plot_experiment_subspace_dimensionality


plot_experiment_theta(
    function_name='Gaussian mixture function', dim='30D',
    method='copy_select', created_at='2019-06-23 08:43:33')

plot_experiment_theta_histogram(
    function_name='Gaussian mixture function', dim='30D',
    method='copy_select', created_at='2019-06-23 08:43:33')

plot_experiment_mask(
    function_name='Gaussian mixture function', dim='30D',
    method='copy_select', created_at='2019-06-23 08:43:33')

plot_experiment_subspace_dimensionality(
    function_name='Gaussian mixture function', dim='30D',
    method='copy_select', created_at='2019-06-23 08:43:33')
