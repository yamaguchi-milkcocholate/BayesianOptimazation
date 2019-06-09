from bayopt.plot.plot import Plot
from matplotlib import pyplot as plt
from bayopt import definitions
from bayopt.clock import clock
from bayopt.utils.utils import mkdir_when_not_exist


class StaticPlot(Plot):

    def __init__(self):
        self.__plt = plt
        self.__figure = plt.figure(figsize=(10, 6))
        self.__plt.cla()

    @classmethod
    def get_plot(cls):
        super().get_plot()

    @classmethod
    def _create_plot(cls):
        return StaticPlot()

    def add_data_set(self, x, y):
        self.__plt.plot(x, y)

    def set_y(self, low_lim, high_lim):
        self.__plt.ylim(low_lim, high_lim)

    def finish(self, option=None):
        mkdir_when_not_exist(abs_path=definitions.ROOT_DIR + '/storage/images')
        if option:
            self.__plt.savefig(definitions.ROOT_DIR + "/storage/images/" + clock.now_str() + '_' + option)
        else:
            self.__plt.savefig(definitions.ROOT_DIR + "/storage/images/" + clock.now_str() + '_' + option)
