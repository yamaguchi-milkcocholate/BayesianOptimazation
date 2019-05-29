from GPyOpt.core.bo import BO
from GPyOpt.core.evaluators import Sequential
from GPyOpt.core.task.cost import CostModel
from GPyOpt.core.task.objective import SingleObjective
from GPyOpt.models.gpmodel import GPModel
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from GPyOpt.optimization.acquisition_optimizer import ContextManager
from GPyOpt.acquisitions.LCB import AcquisitionLCB
from GPyOpt.experiment_design import initial_design
from GPyOpt.util.general import normalize
from GPyOpt.util.duplicate_manager import DuplicateManager
from bayopt.space.space import initialize_space
from bayopt.space.space import get_subspace
from copy import deepcopy
import numpy as np
import time


class Dropout(BO):
    """
    Attributes
        domain (dict | None):
        constraints (dict | None):
        space (Design_space):
        model (BOModel):
        acquisition_optimizer (AcquisitionOptimizer):
        acquisition (AcquisitionBase):

    """

    def __init__(self, f, domain=None, constraints=None, X=None, Y=None, subspace_dim_size=0):

        self.subspace_dim_size = subspace_dim_size
        self.initial_design_numdata = 1
        self.initial_design_type = 'random'
        self.model_type = 'GP'
        self.acquisition_type = 'LCB'
        self.acquisition_optimizer_type = 'lbfgs'
        self.evaluator_type = 'sequential'
        self.cost_withGradients = None
        self.model_update_interval = 1
        self.batch_size = 1
        self.maximize = False
        self.normalize_Y = True
        self.num_cores = 1
        self.verbosity = False
        self.de_duplication = False
        self.exact_feval = False
        self.constraints = None
        self.objective_name = 'no name'

        self.acquisition_optimizer = None
        self.acquisition = None

        self.f = self._sign(f)
        self.objective = SingleObjective(self.f, self.batch_size, 'objective function')
        self.cost = CostModel(cost_withGradients=self.cost_withGradients)

        self.space = initialize_space(domain=domain, constraints=constraints)

        self.set_model()
        # Todo: space=None
        self.set_acquisition_optimizer(space=self.space)
        self.set_acquisition(space=self.space)
        self.set_evaluator(acquisition=self.acquisition)

        self.X = X
        self.Y = Y
        self.set_initial_values()

        super().__init__(
            model=self.model,
            space=self.space,
            objective=self.objective,
            acquisition=self.acquisition,
            evaluator=self.evaluator,
            X_init=self.X,
            Y_init=self.Y,
            cost=self.cost,
            normalize_Y=self.normalize_Y,
            model_update_interval=self.model_update_interval,
            de_duplication=self.de_duplication
        )

    def set_model(self):
        self.model = GPModel(
            kernel=None, noise_var=None, exact_feval=self.exact_feval,
            optimizer='lbfgs', max_iters=1000, optimize_restarts=5,
            sparse=False, num_inducing=10, verbose=False, ARD=False
        )

    def set_acquisition_optimizer(self, space):
        self.acquisition_optimizer = AcquisitionOptimizer(space=space, optimizer='lbfgs')

    def set_acquisition(self, space):
        self.acquisition = AcquisitionLCB(
            model=self.model, space=space, optimizer=self.acquisition_optimizer,
            cost_withGradients=None, exploration_weight=2)

    def set_evaluator(self, acquisition):
        self.evaluator = Sequential(acquisition)

    def set_initial_values(self):
        # Case 1:
        if self.X is None:
            self.X = initial_design(self.initial_design_type, self.space, self.initial_design_numdata)
            self.Y, _ = self.objective.evaluate(self.X)
        # Case 2
        elif self.X is not None and self.Y is None:
            self.Y, _ = self.objective.evaluate(self.X)

    def run_optimization(self, max_iter=0, max_time=np.inf,  eps=1e-8, context=None,
                         verbosity=False, save_models_parameters=True, report_file=None,
                         evaluations_file=None, models_file=None):
        if self.objective is None:
            raise ValueError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.save_models_parameters = save_models_parameters
        self.report_file = report_file
        self.evaluations_file = evaluations_file
        self.models_file = models_file
        self.model_parameters_iterations = None
        self.context = context

        # --- Check if we can save the model parameters in each iteration
        if self.save_models_parameters == True:
            if not (isinstance(self.model, GPModel)):
                print('Models printout after each iteration is only available for GP and GP_MCMC models')
                self.save_models_parameters = False

                # --- Setting up stop conditions
            self.eps = eps
            if (max_iter is None) and (max_time is None):
                self.max_iter = 0
                self.max_time = np.inf
            elif (max_iter is None) and (max_time is not None):
                self.max_iter = np.inf
                self.max_time = max_time
            elif (max_iter is not None) and (max_time is None):
                self.max_iter = max_iter
                self.max_time = np.inf
            else:
                self.max_iter = max_iter
                self.max_time = max_time

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y

        while (self.max_time > self.cum_time):
            # --- update model
            try:
                self._update_model(self.normalization_type)
                self._update_acquisition()
                self._update_evaluator()
            except np.linalg.LinAlgError:
                break

            if (self.num_acquisitions >= self.max_iter
                    or (len(self.X) > 1 and self._distance_last_evaluations() <= self.eps)):
                break

            self.suggested_sample = self._compute_next_evaluations()

            # --- Augment X
            self.X = np.vstack((self.X, self.suggested_sample))

            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1

            if verbosity:
                print("num acquisition: {}, time elapsed: {:.2f}s".format(
                    self.num_acquisitions, self.cum_time))

                # --- Stop messages and execution time
                self._compute_results()

                # --- Print the desired result in files
                if self.report_file is not None:
                    self.save_report(self.report_file)
                if self.evaluations_file is not None:
                    self.save_evaluations(self.evaluations_file)
                if self.models_file is not None:
                    self.save_models(self.models_file)

        # --- Stop messages and execution time
        self._compute_results()

        # --- Print the desired result in files
        if self.report_file is not None:
            self.save_report(self.report_file)
        if self.evaluations_file is not None:
            self.save_evaluations(self.evaluations_file)
        if self.models_file is not None:
            self.save_models(self.models_file)

    def _fill_in_dimensions(self, samples):
        full_num = self.space.objective_dimensionality
        subspace_idx = self.subspace_idx
        embedded_inx = [i for i in range(full_num) if i not in subspace_idx]

        samples_ = list()

        for sample in samples:

            if len(sample) > len(subspace_idx):
                raise ValueError('samples already have been full-dimensionality')

            # Todo: other besides random
            embedded_sample = initial_design('random', self.subspace, 1)[0]

            sample_ = deepcopy(sample)
            for emb_idx, insert_idx in enumerate(embedded_inx):
                sample_ = np.insert(sample_, emb_idx, embedded_sample[emb_idx])

            samples_.append(sample_)

        return np.array(samples_)

    def _update_acquisition(self):
        self.set_acquisition_optimizer(space=self.subspace)
        self.set_acquisition(space=self.subspace)

    def _update_evaluator(self):
        self.evaluator.acquisition = self.acquisition

    def _compute_next_evaluations(self, pending_zipped_X=None, ignored_zipped_X=None):
        ## --- Update the context if any
        self.acquisition.optimizer.context_manager = ContextManager(self.subspace, self.context)

        ### --- Activate de_duplication
        if self.de_duplication:
            duplicate_manager = DuplicateManager(
                space=self.subspace, zipped_X=self.X, pending_zipped_X=pending_zipped_X,
                ignored_zipped_X=ignored_zipped_X)
        else:
            duplicate_manager = None

        ### We zip the value in case there are categorical variables
        suggested_ = self.subspace.zip_inputs(self.evaluator.compute_batch(
            duplicate_manager=duplicate_manager,
            context_manager=self.acquisition.optimizer.context_manager))

        return self._fill_in_dimensions(samples=suggested_)

    def _update_model(self, normalization_type='stats'):
        if self.num_acquisitions % self.model_update_interval == 0:

            self._update_subspace()

            # input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.subspace.unzip_inputs(np.array([xi[[self.subspace_idx]] for xi in self.X]))

            # Y_inmodel is the output that goes into the model
            if self.normalize_Y:
                Y_inmodel = normalize(self.Y, normalization_type)
            else:
                Y_inmodel = self.Y

            self.model.updateModel(X_inmodel, Y_inmodel, None, None)

        # Save parameters of the model
        self._save_model_parameter_values()

    def _update_subspace(self):
        self.subspace_idx = np.random.choice(
            range(self.space.objective_dimensionality),
            self.subspace_dim_size, replace=False)
        self.subspace = get_subspace(space=self.space, subspace_idx=self.subspace_idx)

    def _sign(self, f):
        if self.maximize:
            f_copy = f

            def f(x): return -f_copy(x)
        return f
