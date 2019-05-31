import unittest
import numpy as np
from bayopt.objective_examples.experiments import GaussianMixtureFunction
from bayopt.objective_examples.experiments import gauss


class TestExperiments(unittest.TestCase):

    def test_gaussian_mixture_function(self):
        x = np.array(np.full(30, 2))
        f = GaussianMixtureFunction(dim=len(x))
        print(format(f(x), '.20f'))
        self.assertTrue(isinstance(f(x=x), np.float))
        self.assertTrue(True)

    def test_gauss(self):
        y = gauss(0, 0, 1)
        print(y)


if __name__ == '__main__':
    unittest.main()
