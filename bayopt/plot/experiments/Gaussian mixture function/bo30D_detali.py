from bayopt.plot.utils import plot_experiment_theta_histogram
from bayopt.plot.utils import plot_experiment_theta
from bayopt.plot.utils import plot_experiment_mask
from bayopt.plot.utils import plot_experiment_subspace_dimensionality
from bayopt.plot.utils import plot_experiment_model


function_name = 'Gaussian mixture function'
dim = '30D'
method = 'bo'
created_at = '2019-06-28 09:21:21'


plot_experiment_theta(
    function_name=function_name, dim=dim,
    method=method, created_at=created_at)

plot_experiment_theta_histogram(
    function_name=function_name, dim=dim,
    method=method, created_at=created_at)

plot_experiment_mask(
    function_name=function_name, dim=dim,
    method=method, created_at=created_at)

plot_experiment_subspace_dimensionality(
    function_name=function_name, dim=dim,
    method=method, created_at=created_at)

plot_experiment_model(
    function_name=function_name, dim=dim,
    method=method, created_at=created_at)
