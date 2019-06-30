from GPyOpt.core.evaluators.sequential import Sequential


class SequentialExt(Sequential):

    def __init__(self, acquisition, batch_size=1):
        super(Sequential, self).__init__(acquisition, batch_size)

    def compute_batch(self, duplicate_manager=None,context_manager=None):
        """
        Selects the new location to evaluate the objective.
        """
        x, f = self.acquisition.optimize(duplicate_manager=duplicate_manager)
        return x, f
