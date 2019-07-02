from bayopt.methods.dropout import Dropout
from igo.igo.optimizer.igo import BernoulliIGO
from igo.igo.util.weight import SelectionNonIncFunc
from igo.igo.util.weight import QuantileBasedWeight
from bayopt.space.space import get_subspace
from bayopt.utils.utils import mkdir_when_not_exist
from bayopt.clock.clock import now_str
from bayopt import definitions
from bayopt.methods.evaluator.sequentialext import SequentialExt
import numpy as np


class SelectBase(Dropout):

    UPDATE_EVALUATION = ''

    def __init__(self, fill_in_strategy, f, mix=0.5, domain=None, constraints=None, cost_withGradients=None, X=None, Y=None,
                 model_type='GP', initial_design_numdata=2, initial_design_type='random', acquisition_type='LCB',
                 normalize_Y=True, exact_feval=False, acquisition_optimizer_type='lbfgs', model_update_interval=1,
                 evaluator_type='sequential', batch_size=1, maximize=False, de_duplication=False,
                 sample_num=2, eta=None, theta=None):

        super().__init__(fill_in_strategy=fill_in_strategy, f=f, mix=mix, domain=domain, constraints=constraints,
                         cost_withGradients=cost_withGradients, X=X, Y=Y, subspace_dim_size=None, model_type=model_type,
                         initial_design_numdata=initial_design_numdata, initial_design_type=initial_design_type,
                         acquisition_type=acquisition_type, normalize_Y=normalize_Y, exact_feval=exact_feval,
                         acquisition_optimizer_type=acquisition_optimizer_type,
                         model_update_interval=model_update_interval, evaluator_type=evaluator_type,
                         batch_size=batch_size, maximize=maximize, de_duplication=de_duplication)

        self.sample_num = sample_num

        self._setting_check()

        self.bernoulli_theta = list()
        self.masks = list()
        self.evals = list()

        self.log_masks = list()

        self.sample_index = None
        self.acq_max = None

        # weight function
        non_inc_f = SelectionNonIncFunc(threshold=0.25, negative_weight=True)
        w = QuantileBasedWeight(non_inc_f=non_inc_f, tie_case=True, normalization=False, min_problem=True)

        if theta:
            theta = theta * np.ones(self.dimensionality)
        self.bernoulli_igo = BernoulliIGO(d=self.dimensionality, weight_func=w, eta=eta, theta=theta)

        self.eta = self.bernoulli_igo.eta
        self.theta = self.bernoulli_igo.model.theta[0]

    def _choose_evaluator(self):
        self.evaluator = SequentialExt(self.acquisition)

    def update_subspace(self):
        while True:
            mask = self.sample_mask()[0]
            self.subspace_idx = np.array(np.where(mask == True)[0])

            if len(self.subspace_idx) is not 0:
                self.masks.append(mask)
                self.log_masks.append(mask)
                break

        self.subspace = get_subspace(space=self.space, subspace_idx=self.subspace_idx)

    def _update_distribution(self):
        if len(self.masks) is not self.sample_num:
            raise ValueError('masks are not ' + str(self.sample_num))

        if len(self.evals) is not self.sample_num:
            raise ValueError('evals are not ' + str(self.sample_num))

        self.bernoulli_igo.update(X=np.array(self.masks), evals=np.array(self.evals))
        self._clear_igo_cache()

    def _clear_igo_cache(self):
        self.masks = list()
        self.evals = list()

    def sample_mask(self):
        return self.bernoulli_igo.model.sampling(lam=1)

    def _log_distribution(self):
        self.bernoulli_theta.append(self.bernoulli_igo.model.log())

    def _input_data(self, normalization_type):
        X_inmodel, Y_inmodel = super()._input_data(normalization_type=normalization_type)

        if self.sample_index >= 1:
            X_inmodel = X_inmodel[: -self.sample_index]
            Y_inmodel = Y_inmodel[: -self.sample_index]

        return X_inmodel, Y_inmodel

    def _run_optimization(self):
        while True:

            for i in range(self.sample_num):
                self.sample_index = i
                print('.')

                # --- update model
                try:
                    self.update()

                except np.linalg.LinAlgError:
                    print('np.linalg.LinAlgError')
                    break

                if self.num_acquisitions >= self.max_iter:
                    break

                self.next_point()

                # --- Update current evaluation time and function evaluations
                self.num_acquisitions += 1

            else:
                self._update_distribution()
                self._log_distribution()
                continue

            break

    def _save(self):
        mkdir_when_not_exist(abs_path=definitions.ROOT_DIR + '/storage/' + self.objective_name)
        eta = str('{:.3f}'.format(self.eta)).replace('.', '')
        theta = str('{:.3f}'.format(self.theta)).replace('.', '')

        dir_name = definitions.ROOT_DIR + '/storage/' + self.objective_name + '/' + now_str() + ' ' + str(
            self.dimensionality) + 'D_' + 'e' + eta + 't' + theta + ' ' \
            + str(self.fill_in_strategy) + '_select_' + self.UPDATE_EVALUATION
        mkdir_when_not_exist(abs_path=dir_name)

        self.save_report(report_file=dir_name + '/report.txt')
        self.save_evaluations(evaluations_file=dir_name + '/evaluation.csv')
        self.save_models(models_file=dir_name + '/model.csv')
        self.save_distribution(distribution_file=dir_name + '/distribution.csv')
        self.save_mask(mask_file=dir_name + '/mask.csv')

    def save_distribution(self, distribution_file):
        self._write_csv(distribution_file, self.bernoulli_theta)

    def save_mask(self, mask_file):
        self._write_csv(mask_file, self.log_masks)

    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None):
        context_manager, duplicate_manager = self._compute_setting(pending_zipped_X=pending_zipped_X,
                                                                   ignored_zipped_X=ignored_zipped_X)

        x, self.acq_max = self.evaluator.compute_batch(
            duplicate_manager=duplicate_manager, context_manager=context_manager)

        # We zip the value in case there are categorical variables
        suggested_ = self.subspace.zip_inputs(x)

        return self._fill_in_dimensions(samples=suggested_)

    def _setting_check(self):
        if self.model_update_interval is not 1:
            raise ValueError('model_update_interval != 1')

        if self.initial_design_numdata is not self.sample_num:
            raise ValueError('initial_design_numdata != sample_num')


class SelectObjective(SelectBase):

    UPDATE_EVALUATION = 'objective'

    def next_point(self):
        super().next_point()
        self.evals.append(self.Y_new[0][0])


class SelectAcquisition(SelectBase):

    UPDATE_EVALUATION = 'acquisition'

    def next_point(self):
        super().next_point()
        self.evals.append(self.acq_max[0][0])


class SelectObjectiveDiff(SelectObjective):

    UPDATE_EVALUATION = 'objective_diff'

    def _setting_check(self):
        super()._setting_check()

        if self.sample_num != 2:
            raise ValueError('sample_num must be 2')

    def _update_distribution(self):
        if len(self.masks) is not self.sample_num:
            raise ValueError('masks are not ' + str(self.sample_num))

        if len(self.evals) is not self.sample_num:
            raise ValueError('evals are not ' + str(self.sample_num))

        diff1, diff2 = self.eval_diff()

        self.bernoulli_igo.update(X=np.array(self.masks), evals=np.array([1/diff1, 1/diff2]))
        self._clear_igo_cache()

    def _get_query_points(self):
        return self.X[-2], self.X[-1]

    def eval_diff(self):
        x1, x2 = self._get_query_points()
        f1, f2 = self.evals[0], self.evals[1]
        m1, m2 = self.masks[0], self.masks[1]

        var1, var2 = self.get_cmp_query_point(x1, m1, x2, m2), self.get_cmp_query_point(x2, m2, x1, m1)

        cmp1 = self._evaluate_objective(query=np.array([var1]))
        cmp2 = self._evaluate_objective(query=np.array([var2]))

        return np.abs(f1 - cmp1), np.abs(f2 - cmp2)

    @staticmethod
    def get_cmp_query_point(query, mask, opp_query, opp_mask):
        new = list()
        var_mask = np.logical_not(np.logical_and(np.logical_not(mask), opp_mask))

        for i in range(len(var_mask)):

            if var_mask[i]:
                new.append(opp_query[i])

            else:
                new.append(query[i])

        return np.array(new)

    def _evaluate_objective(self, query):
        y_new, cost_new = self.objective.evaluate(query)
        self.cost.update_cost_model(query, cost_new)

        self.X = np.vstack((self.X, query))
        self.Y = np.vstack((self.Y, y_new))

        self.num_acquisitions += 1

        return y_new[0][0]


class SelectAcquisitionDiff(SelectAcquisition):

    UPDATE_EVALUATION = 'objective_diff'

    def _setting_check(self):
        super()._setting_check()

        if self.sample_num != 2:
            raise ValueError('sample_num must be 2')

    def _update_distribution(self):
        if len(self.masks) is not self.sample_num:
            raise ValueError('masks are not ' + str(self.sample_num))

        if len(self.evals) is not self.sample_num:
            raise ValueError('evals are not ' + str(self.sample_num))

        diff1, diff2 = self.eval_diff()

        self.bernoulli_igo.update(X=np.array(self.masks), evals=np.array([1 / diff1, 1 / diff2]))
        self._clear_igo_cache()

    def _get_query_points(self):
        return self.X[-2], self.X[-1]

    def eval_diff(self):
        x1, x2 = self._get_query_points()
        f1, f2 = self.evals[0], self.evals[1]
        m1, m2 = self.masks[0], self.masks[1]

        var1, var2 = self.get_cmp_query_point(x1, m1, x2, m2), self.get_cmp_query_point(x2, m2, x1, m1)

        cmp1 = self._evaluate_objective(query=np.array([var1]))
        cmp2 = self._evaluate_objective(query=np.array([var2]))

        return np.abs(f1 - cmp1), np.abs(f2 - cmp2)

    @staticmethod
    def get_cmp_query_point(query, mask, opp_query, opp_mask):
        new = list()
        var_mask = np.logical_not(np.logical_and(np.logical_not(mask), opp_mask))

        for i in range(len(var_mask)):

            if var_mask[i]:
                new.append(opp_query[i])

            else:
                new.append(query[i])

        return np.array(new)

    def _evaluate_objective(self, query):
        y_new, cost_new = self.objective.evaluate(query)
        self.cost.update_cost_model(query, cost_new)

        self.X = np.vstack((self.X, query))
        self.Y = np.vstack((self.Y, y_new))

        self.num_acquisitions += 1

        return y_new[0][0]
