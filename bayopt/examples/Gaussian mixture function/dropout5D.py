from bayopt.methods.dropout import Dropout
from bayopt.objective_examples.experiments import GaussianMixtureFunction
import numpy as np

domain = [{'name': 'x0', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          ]

dim = len(domain)

for _ in range(1):

    fill_in_strategy = 'random'
    f = GaussianMixtureFunction(dim=dim, mean_1=2, mean_2=3)
    method = Dropout(
        f=f, domain=domain, subspace_dim_size=2, fill_in_strategy=fill_in_strategy, maximize=True)
    method.run_optimization(max_iter=300)

    fill_in_strategy = 'copy'
    f = GaussianMixtureFunction(dim=dim, mean_1=2, mean_2=3)
    method = Dropout(
        f=f, domain=domain, subspace_dim_size=2, fill_in_strategy=fill_in_strategy, maximize=True)
    method.run_optimization(max_iter=300)

    fill_in_strategy = 'mix'
    f = GaussianMixtureFunction(dim=dim, mean_1=2, mean_2=3)
    method = Dropout(
        f=f, domain=domain, subspace_dim_size=2, fill_in_strategy=fill_in_strategy, maximize=True, mix=0.5)
    method.run_optimization(max_iter=300)
