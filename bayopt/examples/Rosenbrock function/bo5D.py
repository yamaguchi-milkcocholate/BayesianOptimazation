from bayopt.methods.bo import BayesianOptimizationExt
from bayopt.objective_examples.experiments import RosenbrockFunction

domain = [{'name': 'x0', 'type': 'continuous', 'domain': (-2.048, 2.048), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (-2.048, 2.048), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (-2.048, 2.048), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (-2.048, 2.048), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (-2.048, 2.048), 'dimensionality': 1},
          ]

dim = len(domain)
f = RosenbrockFunction(dimensionality=dim)
method = BayesianOptimizationExt(f=f, domain=domain, maximize=False, ard=False)
method.run_optimization(max_iter=100)
