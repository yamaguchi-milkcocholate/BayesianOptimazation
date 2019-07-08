from bayopt.methods.dropout import Dropout
from bayopt.methods.bo import BayesianOptimizationExt
from bayopt.methods.select import *
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

for i in range(1):

    dropout = 5

    """
    dim = len(domain)
    f = MichalewiczFunction(dimensionality=dim, dropout=[i for i in range(dropout)])
    fill_in_strategy = 'copy'
    method = SelectAcquisition(f=f, domain=domain, fill_in_strategy=fill_in_strategy, maximize=False, eta=1 / dim)
    method.run_optimization(max_iter=500)

    dim = len(domain)
    dropout = 5
    f = MichalewiczFunction(dimensionality=dim, dropout=[i for i in range(dropout)])
    fill_in_strategy = 'mix'
    method = SelectAcquisition(f=f, domain=domain, fill_in_strategy=fill_in_strategy, maximize=False, eta=1 / dim)
    method.run_optimization(max_iter=500)
    """

    dim = len(domain)
    f = MichalewiczFunction(dimensionality=dim, dropout=[i for i in range(dropout)])
    method = BayesianOptimizationExt(f=f, domain=domain, maximize=False)
    method.run_optimization(max_iter=500)

    dim = len(domain)
    fill_in_strategy = 'mix'
    f = MichalewiczFunction(dimensionality=dim, dropout=[i for i in range(dropout)])
    method = Dropout(
        f=f, domain=domain, subspace_dim_size=5, fill_in_strategy=fill_in_strategy, maximize=False,
    )
    method.run_optimization(max_iter=500, eps=0)
