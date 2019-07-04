from bayopt.plot.utils import plot_experiment_theta_histogram
from bayopt.plot.utils import plot_experiment_theta
from bayopt.plot.utils import plot_experiment_mask
from bayopt.plot.utils import plot_experiment_subspace_dimensionality
from bayopt.plot.utils import plot_experiment_model
from bayopt.plot.utils import plot_experiment_evaluation


function_name = 'Gaussian mixture function'
dim = '60D_e0017t0500'
method = 'copy_select_acquisition_diff'
created_at = '2019-07-03 00:05:16'


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

plot_experiment_evaluation(
    function_name=function_name, dim=dim,
    method=method, created_at=created_at)
