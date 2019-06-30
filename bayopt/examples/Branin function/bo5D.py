from bayopt.objective_examples.experiments import BraininFunction
from bayopt.methods.bo import BayesianOptimizationExt


domain = [{'name': 'x0', 'type': 'continuous', 'domain': (-5, 15), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (-5, 15), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (-5, 15), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (-5, 15), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (-5, 15), 'dimensionality': 1},
          ]


for i in range(1):

    dim = len(domain)
    f = BraininFunction(1, 3)
    method = BayesianOptimizationExt(f=f, domain=domain, maximize=False, ard=True)
    method.run_optimization(max_iter=250, eps=0)
