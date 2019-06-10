from bayopt.methods.bo import BayesianOptimizationCOMP
from bayopt.objective_examples.experiments import GaussianMixtureFunction
import numpy as np

domain = [{'name': 'x0', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          ]

dim = len(domain)
f = GaussianMixtureFunction(dim=dim, mean_1=2, mean_2=3)
X = np.array([np.full(dim, 1)])
method = BayesianOptimizationCOMP(f=f, domain=domain, maximize=True, X=X)
method.run_optimization(max_iter=300)
