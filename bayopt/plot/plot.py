from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import seaborn as sns
from bayopt import definitions
from bayopt.clock import clock
from bayopt.utils import mkdir_when_not_exist


class Plot(ABC):
    __instance = None

    def __init__(self):
        self.sns = sns
        self.sns.set()
        self._plt = plt
        self.__figure = plt.figure(figsize=(10, 6))
        self._plt.cla()

    @classmethod
    def get_plot(cls):
        if cls.__instance is None:
            cls.__instance = Plot._create_plot()
        return cls.__instance

    @classmethod
    @abstractmethod
    def _create_plot(cls):
        pass

    def finish(self, option=None):
        mkdir_when_not_exist(abs_path=definitions.ROOT_DIR + '/storage/images')
        if option:
            self._plt.savefig(definitions.ROOT_DIR + "/storage/images/" + clock.now_str() + '_' + option)
        else:
            self._plt.savefig(definitions.ROOT_DIR + "/storage/images/" + clock.now_str())

        self._plt.cla()


class StaticPlot(Plot):

    def __init__(self):
        super().__init__()

    @classmethod
    def _create_plot(cls):
        return StaticPlot()

    def add_data_set(self, x, y, label=None):
        if label:
            self._plt.plot(x, y, label=label)
        else:
            self._plt.plot(x, y)

    def add_confidential_area(self, x, upper_confidential_bound, lower_confidential_bound):

        if len(upper_confidential_bound) != len(lower_confidential_bound):
            raise ValueError()

        self._plt.fill_between(x, lower_confidential_bound, upper_confidential_bound, alpha=0.2)

    def set_y(self, low_lim=None, high_lim=None):
        if low_lim is not None and high_lim is not None:
            self._plt.ylim(low_lim, high_lim)

        self._plt.legend(loc='upper left', borderaxespad=0, fontsize=7)


class BarPlot(StaticPlot):

    def __init__(self):
        super().__init__()

    def add_data_set(self, x, y, label=None):
        self._plt.bar(x, y, width=1.0)

    def set_x(self, x, x_ticks):
        self._plt.xticks(x, x_ticks, )


class HeatMap(Plot):

    def __init__(self):
        super().__init__()

    @classmethod
    def _create_plot(cls):
        return StaticPlot()

    def add_data_set(self, data, space):
        if not isinstance(space, tuple):
            raise ValueError('space must be a tuple')

        self.sns.heatmap(data, vmin=space[0], vmax=space[1], cmap='Blues')