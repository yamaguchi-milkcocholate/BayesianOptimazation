from bayopt.plot.utils import plot_experiment_evaluation


function_name = 'Alpine function'
dim = '5D'
method = 'bo'
created_at = '2019-07-05 00:18:01'


plot_experiment_evaluation(
    function_name=function_name, dim=dim,
    method=method, created_at=created_at, maximize=False)
