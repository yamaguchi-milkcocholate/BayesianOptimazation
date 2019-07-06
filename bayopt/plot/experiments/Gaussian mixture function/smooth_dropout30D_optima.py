from bayopt.plot.utils import plot_experiments


plot_experiments(
    function_name='Gaussian mixture function', dim=['30D-5D_smooth', '30D-5D'],
    method=['copy', 'mix'],
    maximize=True, is_median=False, start=None, end=None, dirname='smooth')

plot_experiments(
    function_name='Gaussian mixture function', dim=['30D-5D_smooth', '30D-5D'],
    method=['copy', 'mix'],
    maximize=True, is_median=True, start=None, end=None,  dirname='smooth')
