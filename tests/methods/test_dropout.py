import unittest
from bayopt.methods.dropout import Dropout
import numpy as np


class TestDropout(unittest.TestCase):

    def setUp(self) -> None:
        def f(x):
            x0, x1 = x[:, 0], x[:, 1]
            f0 = np.log(10.5 - x0) + 0.1 * np.sin(15 * x0)
            f1 = np.cos(1.5 * x0) + 0.1 * x0
            return (1 - x1) * f0 + x1 * f1

        bounds = [{'name': 'x0', 'type': 'continuous', 'domain': (0, 10), 'dimensionality': 1},
                  {'name': 'x1', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 1},
                  {'name': 'x2', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 1}]
        self.method = Dropout(
            f=f, domain=bounds, subspace_dim_size=2
        )

    def test_example(self):
        self.method.run_optimization(max_iter=30)
        self.assertTrue(True)

    def test_check_domain(self):
        domain = self.method.space.config_space
        self.assertEqual(domain[0]['name'], '0')
        self.assertEqual(domain[1]['name'], '1')
        self.assertEqual(domain[2]['name'], '2')

    def test_subspace(self):
        space = self.method.space
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
