import GPyOpt
import numpy as np


def f(x):
    x0, x1 = x[:, 0], x[:, 1]
    f0 = np.log(10.5-x0) + 0.1*np.sin(15*x0)
    f1 = np.cos(1.5*x0) + 0.1*x0
    return (1-x1)*f0 + x1*f1


bounds = [{'name': 'x0', 'type': 'continuous', 'domain': (0, 10), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 2},
          {'name': 'x2', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 3}]

myBopt = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)

myBopt.run_optimization(max_iter=30)
