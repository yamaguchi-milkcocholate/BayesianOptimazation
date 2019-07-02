from bayopt.methods.select import SelectAcquisitionDiff
from bayopt.objective_examples.experiments import GaussianMixtureFunction

domain = [{'name': 'x0', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          ]


for i in range(1):

    dim = len(domain)
    fill_in_strategy = 'random'
    f = GaussianMixtureFunction(dim=dim, mean_1=2, mean_2=3)
    method = SelectAcquisitionDiff(f=f, domain=domain, fill_in_strategy=fill_in_strategy, maximize=True)
    # method.run_optimization(max_iter=500, eps=0)

    dim = len(domain)
    fill_in_strategy = 'copy'
    f = GaussianMixtureFunction(dim=dim, mean_1=2, mean_2=3)
    method = SelectAcquisitionDiff(f=f, domain=domain, fill_in_strategy=fill_in_strategy, maximize=True)
    method.run_optimization(max_iter=50, eps=0)

    dim = len(domain)
    fill_in_strategy = 'mix'
    f = GaussianMixtureFunction(dim=dim, mean_1=2, mean_2=3)
    method = SelectAcquisitionDiff(f=f, domain=domain, fill_in_strategy=fill_in_strategy, maximize=True, mix=0.5)
    # method.run_optimization(max_iter=500, eps=0)
