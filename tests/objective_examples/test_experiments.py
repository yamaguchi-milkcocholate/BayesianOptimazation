import unittest
import numpy as np
from bayopt.objective_examples.experiments import GaussianMixtureFunction


class TestExperiments(unittest.TestCase):

    def test_gaussian_mixture_function(self):
        x = np.array([1, 1, 1])
        f = GaussianMixtureFunction(dim=len(x))

        self.assertTrue(isinstance(f(x=x), np.float))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
