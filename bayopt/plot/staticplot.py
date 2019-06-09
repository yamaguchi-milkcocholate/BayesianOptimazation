from bayopt.plot.plot import Plot
from matplotlib import pyplot as plt
from bayopt import definitions
from bayopt.clock import clock


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

    def finish(self):

        self.__plt.savefig(definitions.ROOT_DIR + "/storage/images/" + clock.now_str())
