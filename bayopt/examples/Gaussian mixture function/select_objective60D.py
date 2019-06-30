from bayopt.methods.select import SelectObjective
from bayopt.objective_examples.experiments import GaussianMixtureFunction

domain = [{'name': 'x0', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x5', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x6', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x7', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x8', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x9', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x10', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x11', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x12', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x13', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x14', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x15', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x16', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x17', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x18', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x19', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x20', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x21', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x22', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x23', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x24', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x25', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x26', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x27', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x28', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x29', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x30', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x31', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x32', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x33', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x34', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x35', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x36', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x37', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x38', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x39', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x40', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x41', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x42', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x43', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x44', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x45', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x46', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x47', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x48', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x49', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x50', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x51', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x52', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x53', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x54', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x55', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x56', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x57', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x58', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          {'name': 'x59', 'type': 'continuous', 'domain': (1, 4), 'dimensionality': 1},
          ]


for i in range(5):

    dim = len(domain)
    fill_in_strategy = 'random'
    f = GaussianMixtureFunction(dim=dim, mean_1=2, mean_2=3)
    method = SelectObjective(f=f, domain=domain, fill_in_strategy=fill_in_strategy, maximize=True)
    method.run_optimization(max_iter=500, eps=0)

    dim = len(domain)
    fill_in_strategy = 'copy'
    f = GaussianMixtureFunction(dim=dim, mean_1=2, mean_2=3)
    method = SelectObjective(f=f, domain=domain, fill_in_strategy=fill_in_strategy, maximize=True)
    method.run_optimization(max_iter=500, eps=0)

    dim = len(domain)
    fill_in_strategy = 'mix'
    f = GaussianMixtureFunction(dim=dim, mean_1=2, mean_2=3)
    method = SelectObjective(f=f, domain=domain, fill_in_strategy=fill_in_strategy, maximize=True, mix=0.5)
    method.run_optimization(max_iter=500, eps=0)
