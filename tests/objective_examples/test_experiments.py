import unittest
import numpy as np
from bayopt.objective_examples.experiments import GaussianMixtureFunction
from bayopt.objective_examples.experiments import SchwefelsFunction


class TestExperiments(unittest.TestCase):

    def test_gaussian_mixture_function(self):
        x = np.array(np.full(30, 2))
        f = GaussianMixtureFunction(dim=len(x), mean_1=2, mean_2=3)
        self.assertTrue(isinstance(f(x=x), np.float))
        self.assertEqual(1, f(x=x))

    def test_schwefels_function(self):
        x = np.array(np.zeros(30))
        f = SchwefelsFunction()
        self.assertTrue(isinstance(f(x=x), np.float))
        self.assertEqual(0, f(np.zeros(30)))
        self.assertEqual(9455, f(np.ones(30)))


if __name__ == '__main__':
    unittest.main()
