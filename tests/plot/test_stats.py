import unittest
import numpy as np
from bayopt.plot.stats import maximum_locus
from bayopt.plot.stats import minimum_locus
from bayopt.plot.stats import with_confidential


class TestStats(unittest.TestCase):

    def test_maximum_locus(self):
        data = np.array([1, 2, 4, 3])

        results = maximum_locus(data)

        self.assertTrue(np.all(np.array([1, 2, 4, 4]) == results))

    def test_minimum_locus(self):
        data = np.array([3, 4, 2, 1])

        results = minimum_locus(data)

        self.assertTrue(np.all(np.array([3, 3, 2, 1]) == results))

    def test_maximum_locus_shape_exception(self):
        data = np.array([[1, 2], [3, 4]])

        with self.assertRaises(ValueError):
            results = maximum_locus(data)

        pass

    def test_minimum_locus_shape_exception(self):
        data = np.array([[1, 2], [3, 4]])

        with self.assertRaises(ValueError):
            results = minimum_locus(data)

    def test_with_confidential(self):
        data = np.array([[1, 3, 2], [2, 4, 3]])

        results = with_confidential(data)

        self.assertEqual(results.values.shape[0], 2)
        self.assertEqual(results.values.shape[1], 5)
