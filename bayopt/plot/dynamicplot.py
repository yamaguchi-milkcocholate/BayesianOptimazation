from bayopt.plot.plot import Plot
from matplotlib import pyplot as plt


class DynamicPlot(Plot):

    @classmethod
    def get_plot(cls):
        super().get_plot()

    @classmethod
    def _create_plot(cls):
        return DynamicPlot()
