from bayopt.objective_examples.experiments import GaussianMixtureFunction
from bayopt.methods.rembo import REMBO
import numpy as np


for i in range(5):

    n_dims = 30
    data_space = np.array([[1, 4] for j in range(n_dims)])
    f = GaussianMixtureFunction(dim=n_dims, mean_1=2, mean_2=3)
    n_embedding_dims = 5
    n_repetitions = 10
    n_trials = 100
    kappa = 2.5

    method = REMBO(f=f, n_dims=n_dims, n_embedding_dims=n_embedding_dims, data_space=data_space)
    method.run_optimization(max_iter=500)
