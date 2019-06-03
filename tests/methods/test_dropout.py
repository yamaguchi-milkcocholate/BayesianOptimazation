import unittest
from bayopt.methods.dropout import Dropout
import numpy as np

"""
class TestDropout(unittest.TestCase):

    def setUp(self) -> None:
        def f(x):
            x0, x1 = x[:, 0], x[:, 1]
            return -0.1*(x0**2+x1**2-16)**2 + 10*np.sin(3*x0)

        bounds = [{'name': 'x0', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
                  {'name': 'x1', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
                  {'name': 'x2', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
                  {'name': 'x3', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
                  {'name': 'x4', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
                  ]
        self.method = Dropout(
            f=f, domain=bounds, subspace_dim_size=3, fill_in_strategy='random'
        )

    def test_check_domain(self):
        domain = self.method.space.config_space
        self.assertEqual(domain[0]['name'], '0')
        self.assertEqual(domain[1]['name'], '1')
        self.assertEqual(domain[2]['name'], '2')
        self.assertEqual(domain[3]['name'], '3')
        self.assertEqual(domain[4]['name'], '4')
"""

if __name__ == '__main__':
    unittest.main()
