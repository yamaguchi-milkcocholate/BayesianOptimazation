from bayopt.methods.dropout import Dropout
from igo.igo.optimizer.igo import BernoulliIGO
from igo.igo.util.weight import SelectionNonIncFunc
from igo.igo.util.weight import QuantileBasedWeight
from bayopt.space.space import get_subspace
from bayopt.utils.utils import mkdir_when_not_exist
from bayopt.clock.clock import now_str
from bayopt import definitions
import numpy as np


class Select(Dropout):

    def __init__(self, fill_in_strategy, f, mix=0.5, domain=None, constraints=None, cost_withGradients=None, X=None, Y=None,
                 model_type='GP', initial_design_numdata=2, initial_design_type='random', acquisition_type='LCB',
                 normalize_Y=True, exact_feval=False, acquisition_optimizer_type='lbfgs', model_update_interval=1,
                 evaluator_type='sequential', batch_size=1, maximize=False, de_duplication=False, sample_num=2):

        if initial_design_numdata is not sample_num:
            raise ValueError('initial_design_numdata != sample_num')

        if model_update_interval is not 1:
            raise ValueError('model_update_interval != 1')

        super().__init__(fill_in_strategy=fill_in_strategy, f=f, mix=mix, domain=domain, constraints=constraints,
                         cost_withGradients=cost_withGradients, X=X, Y=Y, subspace_dim_size=None, model_type=model_type,
                         initial_design_numdata=initial_design_numdata, initial_design_type=initial_design_type,
                         acquisition_type=acquisition_type, normalize_Y=normalize_Y, exact_feval=exact_feval,
                         acquisition_optimizer_type=acquisition_optimizer_type,
                         model_update_interval=model_update_interval, evaluator_type=evaluator_type,
                         batch_size=batch_size, maximize=maximize, de_duplication=de_duplication)

        self.sample_num = sample_num
        self.bernoulli_theta = list()

        # weight function
        non_inc_f = SelectionNonIncFunc(threshold=0.25, negative_weight=True)
        w = QuantileBasedWeight(non_inc_f=non_inc_f, tie_case=True, normalization=False, min_problem=False)

        self.bernoulli_igo = BernoulliIGO(d=self.dimensionality, weight_func=w)

    def update_subspace(self):
        while True:
            mask = self.sample_mask()[0]
            self.subspace_idx = np.array(np.where(mask == True)[0])

            if len(self.subspace_idx) is not 0:
                break

        self.subspace = get_subspace(space=self.space, subspace_idx=self.subspace_idx)

    def _update_mask_distribution(self):
        if self.Y_new.shape is tuple([self.sample_num, 1]):
            raise ValueError('Y_new must be a 2d array-like')

        if self.suggested_sample is tuple([self.sample_num, self.dimensionality]):
            raise ValueError('suggested_sample must be a 2d array-like')

        self.bernoulli_igo.update(X=self.suggested_sample, evals=self.Y_new[:, 0])

    def sample_mask(self):
        return self.bernoulli_igo.model.sampling(lam=1)

    def _log_distribution(self):
        self.bernoulli_theta.append(self.bernoulli_igo.model.log())

    def _run_optimization(self):
        while True:

            for i in range(self.sample_num):
                print('.')

                # --- update model
                try:
                    self.update()

                except np.linalg.LinAlgError:
                    print('np.linalg.LinAlgError')
                    break

                # --- Update current evaluation time and function evaluations
                self.num_acquisitions += 1

            if self.num_acquisitions >= self.max_iter:
                break

            self._update_mask_distribution()
            self._log_distribution()

    def _save(self):
        mkdir_when_not_exist(abs_path=definitions.ROOT_DIR + '/storage/' + self.objective_name)

        dir_name = definitions.ROOT_DIR + '/storage/' + self.objective_name + '/' + now_str() + ' ' + str(
            self.dimensionality) + 'D ' + str(self.fill_in_strategy) + ' select'
        mkdir_when_not_exist(abs_path=dir_name)

        self.save_report(report_file=dir_name + '/report.txt')
        self.save_evaluations(evaluations_file=dir_name + '/evaluation.csv')
        self.save_models(models_file=dir_name + '/model.csv')
        self.save_distribution(distribution_file=dir_name + '/distribution.csv')

    def save_distribution(self, distribution_file):
        self._write_csv(distribution_file, self.bernoulli_theta)
