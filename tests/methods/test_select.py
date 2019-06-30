import unittest
from bayopt.methods.select import SelectObjective
from tests.utils.example_function import ExampleFunction


class TestSelect(unittest.TestCase):

    def setUp(self) -> None:
        self.f = ExampleFunction()

        self.domain = [
            {'name': 'x0', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
            {'name': 'x1', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
            {'name': 'x2', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
            {'name': 'x3', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
            {'name': 'x4', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
        ]

    def test_example(self):
        method = SelectObjective(fill_in_strategy='random', f=self.f, domain=self.domain)
        method.run_optimization(max_iter=10)
        self.assertEqual(method.num_acquisitions, 10)
        self.assertEqual(len(method.bernoulli_theta), 4)
        self.assertTrue(True)
