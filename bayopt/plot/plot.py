from abc import ABC, abstractmethod


class Plot(ABC):
    __instance = None

    @classmethod
    def get_plot(cls):
        if cls.__instance is None:
            cls.__instance = Plot._create_plot()
        return cls.__instance

    @classmethod
    @abstractmethod
    def _create_plot(cls):
        pass
