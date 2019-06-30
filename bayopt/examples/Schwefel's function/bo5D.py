from bayopt.methods.bo import BayesianOptimizationExt
from bayopt.objective_examples.experiments import SchwefelsFunction
import numpy as np

domain = [{'name': 'x0', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          ]

dim = len(domain)
f = SchwefelsFunction()
method = BayesianOptimizationExt(f=f, domain=domain, maximize=False)
method.run_optimization(max_iter=100)
