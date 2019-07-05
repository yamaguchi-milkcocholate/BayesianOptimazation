from bayopt.methods.bo import BayesianOptimizationExt
from bayopt.objective_examples.experiments import MichalewiczFunction
import numpy as np

domain = [{'name': 'x0', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          ]

for _ in range(1):
    dim = len(domain)
    f = MichalewiczFunction(dimensionality=dim)
    method = BayesianOptimizationExt(f=f, domain=domain, maximize=False, ard=False)
    method.run_optimization(max_iter=500)
