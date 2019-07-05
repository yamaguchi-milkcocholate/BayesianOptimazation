import unittest
import numpy as np
from bayopt.objective_examples.experiments import GaussianMixtureFunction
from bayopt.objective_examples.experiments import SchwefelsFunction
from bayopt.objective_examples.experiments import AlpineFunction
from bayopt.objective_examples.experiments import RosenbrockFunction
from bayopt.objective_examples.experiments import MichalewiczFunction


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

        x = np.array([1, 2, 3, 4])
        self.assertEqual(146, f(x))

    def test_alpine_function(self):
        x = np.array([1, 1, 1])
        f = AlpineFunction(dimensionality=len(x))
        x = np.array([0, 0, 0])
        self.assertEqual(0, f(x))

        x = np.array([1, 1, 1])
        f = AlpineFunction(dimensionality=len(x), dropout=[0, 2])
        x = np.array([0, 0, 0])
        self.assertEqual(0, f(x))

    def test_brosenrock_function(self):
        x = np.array([1, 2])
        f = RosenbrockFunction(dimensionality=2)
        self.assertEqual(0, f.sub_func(1, 1))
        self.assertEqual(100, f.sub_func(1, 2))

        self.assertEqual(100, f(x))

    def test_michalewecz_function(self):
        x = np.array([2.20, 1.57])
        f = MichalewiczFunction(dimensionality=2, m=10)
        self.assertEqual('{:.3f}'.format(-1.8013), '{:.3f}'.format(f(x)))


if __name__ == '__main__':
    unittest.main()
