from bayopt.objective_examples.experiments import BraininFunction
from bayopt.methods.rembo import REMBO

f = BraininFunction()

n_dims = 10**3
n_embedding_dims = 4
n_repetitions = 10
n_trials = 100
kappa = 2.5

method = REMBO(f=f, n_dims=n_dims, n_embedding_dims=n_embedding_dims)
method.run_optimization(max_iter=500)
