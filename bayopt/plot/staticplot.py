from bayopt.plot.plot import Plot


class StaticPlot(Plot):

    def __init__(self):
        super().__init__()

    @classmethod
    def get_plot(cls):
        super().get_plot()

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

        self._plt.fill_between(x, lower_confidential_bound, upper_confidential_bound, alpha=0.25)

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
    def get_plot(cls):
        super().get_plot()

    @classmethod
    def _create_plot(cls):
        return StaticPlot()

    def add_data_set(self, data, space):
        if not isinstance(space, tuple):
            raise ValueError('space must be a tuple')

        self.sns.heatmap(data, vmin=space[0], vmax=space[1], cmap='Blues')
