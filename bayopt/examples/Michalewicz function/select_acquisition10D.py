from bayopt.methods.select import SelectAcquisition
from bayopt.objective_examples.experiments import MichalewiczFunction
import numpy as np

domain = [{'name': 'x0', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x5', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x6', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x7', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x8', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          {'name': 'x9', 'type': 'continuous', 'domain': (0, np.pi), 'dimensionality': 1},
          ]


for i in range(5):

    dim = len(domain)
    fill_in_strategy = 'copy'
    f = MichalewiczFunction(dimensionality=dim)
    method = SelectAcquisition(f=f, domain=domain, fill_in_strategy=fill_in_strategy, maximize=False, eta=1 / dim)
    method.run_optimization(max_iter=500, eps=0)

    dim = len(domain)
    fill_in_strategy = 'mix'
    f = MichalewiczFunction(dimensionality=dim)
    method = SelectAcquisition(f=f, domain=domain, fill_in_strategy=fill_in_strategy, maximize=False, eta=1 / dim)
    method.run_optimization(max_iter=500, eps=0)
