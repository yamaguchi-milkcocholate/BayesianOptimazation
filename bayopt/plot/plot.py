from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import seaborn as sns
from bayopt import definitions
from bayopt.clock import clock
from bayopt.utils.utils import mkdir_when_not_exist


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
