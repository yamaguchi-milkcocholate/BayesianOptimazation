from bayopt.methods.bo import BayesianOptimizationCOMP
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
X = np.array([np.full(dim, 1)])
method = BayesianOptimizationCOMP(f=f, domain=domain, maximize=False, X=X)
method.run_optimization(max_iter=300)
