from bayopt.plot.utils import plot_experiments


plot_experiments(
    function_name='Gaussian mixture function', dim=['10D-2D_smooth'],
    method=['copy'],
    maximize=True, is_median=False, start=None, end=None, dirname=None)

plot_experiments(
    function_name='Gaussian mixture function', dim=['10D-2D_smooth'],
    method=['copy'],
    maximize=True, is_median=True, start=None, end=None,  dirname=None)
