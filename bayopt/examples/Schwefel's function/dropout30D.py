from bayopt.methods.dropout import Dropout
from bayopt.objective_examples.experiments import SchwefelsFunction

domain = [{'name': 'x0', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x3', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x4', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x5', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x6', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x7', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x8', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x9', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x10', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x11', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x12', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x13', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x14', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x15', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x16', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x17', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x18', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x19', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x20', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x21', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x22', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x23', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x24', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x25', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x26', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x27', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x28', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          {'name': 'x29', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
          ]


for i in range(5):

    fill_in_strategy = 'random'
    f = SchwefelsFunction()
    method = Dropout(f=f, domain=domain, subspace_dim_size=5, fill_in_strategy=fill_in_strategy, maximize=False)
    method.run_optimization(max_iter=500, eps=0)

    fill_in_strategy = 'copy'
    f = SchwefelsFunction()
    method = Dropout(f=f, domain=domain, subspace_dim_size=1, fill_in_strategy=fill_in_strategy, maximize=False,
                     )
    method.run_optimization(max_iter=500, eps=0)

    fill_in_strategy = 'mix'
    f = SchwefelsFunction()
    method = Dropout(f=f, domain=domain, subspace_dim_size=5, fill_in_strategy=fill_in_strategy, maximize=False, mix=0.5)
    method.run_optimization(max_iter=500, eps=0)
