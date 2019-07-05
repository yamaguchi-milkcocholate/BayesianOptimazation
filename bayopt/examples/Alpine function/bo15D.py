from bayopt.methods.bo import BayesianOptimizationExt
from bayopt.objective_examples.experiments import AlpineFunction

domain = [{'name': 'x0', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x5', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x6', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x7', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x8', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x9', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x10', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x11', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x12', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x13', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          {'name': 'x14', 'type': 'continuous', 'domain': (-10, 10), 'dimensionality': 1},
          ]

for i in range(1):

    dim = len(domain)
    f = AlpineFunction(dimensionality=dim)
    method = BayesianOptimizationExt(f=f, domain=domain, maximize=False, ard=False)
    method.run_optimization(max_iter=500)
