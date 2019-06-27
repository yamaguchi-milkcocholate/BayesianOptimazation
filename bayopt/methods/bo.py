from GPyOpt.methods.bayesian_optimization import BayesianOptimization
from GPyOpt.core.errors import InvalidConfigError
from bayopt.clock.clock import now_str
from bayopt import definitions
from bayopt.utils.utils import mkdir_when_not_exist
import GPyOpt
import numpy as np
import time
from copy import deepcopy


class BayesianOptimizationExt(BayesianOptimization):

    def __init__(self, f, domain=None, constraints=None, cost_withGradients=None, model_type='GP',
                 X=None, Y=None, initial_design_numdata=1, initial_design_type='random', acquisition_type='LCB',
                 normalize_Y=True, exact_feval=False, acquisition_optimizer_type='lbfgs', model_update_interval=1,
                 evaluator_type='sequential', batch_size=1, num_cores=1, verbosity=False, verbosity_model=False,
                 maximize=False, de_duplication=False, ard=False):

        super(BayesianOptimizationExt, self).__init__(
            f, domain=domain, constraints=constraints, cost_withGradients=cost_withGradients,
            model_type=model_type, X=X, Y=Y, initial_design_numdata=initial_design_numdata,
            initial_design_type=initial_design_type, acquisition_type=acquisition_type, normalize_Y=normalize_Y,
            exact_feval=exact_feval, acquisition_optimizer_type=acquisition_optimizer_type,
            model_update_interval=model_update_interval, evaluator_type=evaluator_type,
            batch_size=batch_size, num_cores=num_cores, verbosity=verbosity, verbosity_model=verbosity_model,
            maximize=maximize, de_duplication=de_duplication, ARD=ard)

        self.objective_name = f.get_function_name()
        self.ard = ard

    def run_optimization(self, max_iter=0, max_time=np.inf,  eps=1e-8, context=None, verbosity=False, save_models_parameters=True, report_file=None, evaluations_file=None, models_file=None):
        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

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
            if not (isinstance(self.model, GPyOpt.models.GPModel) or isinstance(self.model, GPyOpt.models.GPModel_MCMC)):
                print('Models printout after each iteration is only available for GP and GP_MCMC models')
                self.save_models_parameters = False

        # --- Setting up stop conditions
        self.eps = eps
        if  (max_iter is None) and (max_time is None):
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

        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y

        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time):
            print('.')

            # --- Update model
            try:
                self._update_model(self.normalization_type)
            except np.linalg.linalg.LinAlgError:
                break

            if (self.num_acquisitions >= self.max_iter
                    or (len(self.X) > 1 and self._distance_last_evaluations() <= self.eps)):
                break

            self.suggested_sample = self._compute_next_evaluations()

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))

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

        self._save()

    def _init_design_chooser(self):
        super()._init_design_chooser()

        # save initial values
        self.initial_X = deepcopy(self.X)
        if self.maximize:
            self.initial_Y = -deepcopy(self.Y)
        else:
            self.initial_Y = deepcopy(self.Y)

    def _save(self):
        mkdir_when_not_exist(abs_path=definitions.ROOT_DIR + '/storage/' + self.objective_name)

        dir_name = definitions.ROOT_DIR + '/storage/' + self.objective_name + '/' + now_str() + ' ' + str(self.space.dimensionality) + 'D bo'
        mkdir_when_not_exist(abs_path=dir_name)

        self.save_report(report_file=dir_name + '/report.txt')
        self.save_evaluations(evaluations_file=dir_name + '/evaluation.csv')
        self.save_models(models_file=dir_name + '/model.csv')

    def save_report(self, report_file= None):
        with open(report_file,'w') as file:
            import GPyOpt
            import time

            file.write('-----------------------------' + ' GPyOpt Report file ' + '-----------------------------------\n')
            file.write('GPyOpt Version ' + str(GPyOpt.__version__) + '\n')
            file.write('Date and time:               ' + time.strftime("%c")+'\n')
            if self.num_acquisitions==self.max_iter:
                file.write('Optimization completed:      ' +'YES, ' + str(self.X.shape[0]).strip('[]') + ' samples collected.\n')
                file.write('Number initial samples:      ' + str(self.initial_design_numdata) +' \n')
            else:
                file.write('Optimization completed:      ' +'NO,' + str(self.X.shape[0]).strip('[]') + ' samples collected.\n')
                file.write('Number initial samples:      ' + str(self.initial_design_numdata) +' \n')

            file.write('Tolerance:                   ' + str(self.eps) + '.\n')
            file.write('Optimization time:           ' + str(self.cum_time).strip('[]') +' seconds.\n')

            file.write('\n')
            file.write('--------------------------------' + ' Problem set up ' + '------------------------------------\n')
            file.write('Problem name:                ' + self.objective_name +'\n')
            file.write('Problem dimension:           ' + str(self.space.dimensionality) +'\n')
            file.write('Number continuous variables  ' + str(len(self.space.get_continuous_dims()) ) +'\n')
            file.write('Number discrete variables    ' + str(len(self.space.get_discrete_dims())) +'\n')
            file.write('Number bandits               ' + str(self.space.get_bandit().shape[0]) +'\n')
            file.write('Noiseless evaluations:       ' + str(self.exact_feval) +'\n')
            file.write('Cost used:                   ' + self.cost.cost_type +'\n')
            file.write('Constraints:                  ' + str(self.constraints==True) +'\n')

            file.write('\n')
            file.write('------------------------------' + ' Optimization set up ' + '---------------------------------\n')
            file.write('Normalized outputs:          ' + str(self.normalize_Y) + '\n')
            file.write('Model type:                  ' + str(self.model_type).strip('[]') + '\n')
            file.write('Model update interval:       ' + str(self.model_update_interval) + '\n')
            file.write('Acquisition type:            ' + str(self.acquisition_type).strip('[]') + '\n')
            file.write('Acquisition optimizer:       ' + str(self.acquisition_optimizer.optimizer_name).strip('[]') + '\n')

            file.write('Acquisition type:            ' + str(self.acquisition_type).strip('[]') + '\n')
            if hasattr(self, 'acquisition_optimizer') and hasattr(self.acquisition_optimizer, 'optimizer_name'):
                file.write('Acquisition optimizer:       ' + str(self.acquisition_optimizer.optimizer_name).strip('[]') + '\n')
            else:
                file.write('Acquisition optimizer:       None\n')
            file.write('Evaluator type (batch size): ' + str(self.evaluator_type).strip('[]') + ' (' + str(self.batch_size) + ')' + '\n')
            file.write('Cores used:                  ' + str(self.num_cores) + '\n')
            file.write('ARD used:                    ' + str(self.ard) + '\n')

            file.write('\n')
            file.write('---------------------------------' + ' Summary ' + '------------------------------------------\n')
            file.write('Initial X:                       ' + str(self.initial_X) + '\n')
            file.write('Initial Y:                       ' + str(self.initial_Y) + '\n')
            if self.maximize:
                file.write('Value at maximum:            ' + str(format(-min(self.Y)[0], '.20f')).strip('[]') +'\n')
                file.write('Best found maximum location: ' + str(self.X[np.argmin(self.Y),:]).strip('[]') +'\n')
            else:
                file.write('Value at minimum:            ' + str(format(min(self.Y)[0], '.20f')).strip('[]') +'\n')
                file.write('Best found minimum location: ' + str(self.X[np.argmin(self.Y),:]).strip('[]') +'\n')

            file.write('----------------------------------------------------------------------------------------------\n')
            file.close()
