import unittest
import numpy as np
from bayopt.objective_examples.experiments import GaussianMixtureFunction


class TestExperiments(unittest.TestCase):

    def test_gaussian_mixture_function(self):
        x = np.array(np.full(30, 2))
        f = GaussianMixtureFunction(dim=len(x), mean_1=2, mean_2=3)
        self.assertTrue(isinstance(f(x=x), np.float))
        self.assertEqual(1, f(x=x))


if __name__ == '__main__':
    unittest.main()
