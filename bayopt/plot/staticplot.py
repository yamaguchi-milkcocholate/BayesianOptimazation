from bayopt.plot.plot import Plot
from matplotlib import pyplot as plt
import seaborn as sns
from bayopt import definitions
from bayopt.clock import clock
from bayopt.utils.utils import mkdir_when_not_exist


class StaticPlot(Plot):

    def __init__(self):
        self.sns = sns
        self.sns.set()
        self.__plt = plt
        self.__figure = plt.figure(figsize=(10, 6))
        self.__plt.cla()

    @classmethod
    def get_plot(cls):
        super().get_plot()

    @classmethod
    def _create_plot(cls):
        return StaticPlot()

    def add_data_set(self, x, y, label=None):
        if label:
            self.__plt.plot(x, y, label=label)
        else:
            self.__plt.plot(x, y)

    def add_confidential_area(self, x, mean, std):
        if len(mean) != len(std):
            raise ValueError()

        self.__plt.fill_between(x, mean - std, mean + std, alpha=0.25)

    def set_y(self, low_lim=None, high_lim=None):
        if low_lim is not None and high_lim is not None:
            self.__plt.ylim(low_lim, high_lim)

        self.__plt.legend(loc='upper left')

    def finish(self, option=None):
        mkdir_when_not_exist(abs_path=definitions.ROOT_DIR + '/storage/images')
        if option:
            self.__plt.savefig(definitions.ROOT_DIR + "/storage/images/" + clock.now_str() + '_' + option)
        else:
            self.__plt.savefig(definitions.ROOT_DIR + "/storage/images/" + clock.now_str())
