from bayesian_optimization.bayesian_optimization.bayesian_optimization import REMBOOptimizer
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from bayesian_optimization.bayesian_optimization.model import GaussianProcessModel
from bayesian_optimization.bayesian_optimization.acquisition_functions import UpperConfidenceBound
from bayopt import definitions
from bayopt.clock.clock import now_str
from bayopt.utils.utils import mkdir_when_not_exist
import numpy as np
import pickle


class REMBO(REMBOOptimizer):

    def __init__(self, f, n_dims, n_embedding_dims=2, data_space=None,
                 n_keep_dims=0):
        self.kappa = 2.5
        self.n_dims = n_dims

        kernel = C(1.0, (0.01, 1000.0)) \
                 * Matern(length_scale=1.0, length_scale_bounds=[(0.001, 100)])
        self.model = GaussianProcessModel(kernel=kernel)
        self.acquisition_function = UpperConfidenceBound(self.model, kappa=self.kappa)

        super().__init__(n_dims=n_dims, n_embedding_dims=n_embedding_dims, data_space=data_space,
                         n_keep_dims=n_keep_dims, model=self.model, acquisition_function=self.acquisition_function)
        self.f = f
        self.objective_name = self.f.get_function_name()

    def run_optimization(self, max_iter):
        for i in range(max_iter):
            print('.')
            X_query = \
                self.select_query_point(boundaries=np.array([[-5, 15]] * self.n_dims))
            y_query = self.f(X_query)
            self.update(X_query, y_query)

        self._save()

    def _save(self):
        mkdir_when_not_exist(abs_path=definitions.ROOT_DIR + '/storage/' + self.objective_name)

        dir_name = definitions.ROOT_DIR + '/storage/' + self.objective_name + '/' + now_str() + ' ' + str(
            self.n_dims) + 'D EM' + str(self.n_embedding_dims)
        mkdir_when_not_exist(dir_name)

        self._save_report(report_file=dir_name + '/report.txt')
        self._save_self(file=dir_name + '/self.pickle')

    def _save_report(self, report_file):
        with open(report_file, 'w') as file:
            file.write('---------------------------------' + ' Summary ' + '------------------------------------------\n')
            file.write('Value at maximum:            ' + str(format(self.best_value(), '.20f')).strip('[]') + '\n')
            file.write('Best found maximum location: ' + str(self.best_params()).strip('[]') + '\n')
            file.close()

    def _save_self(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)
