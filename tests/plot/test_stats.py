import unittest
import numpy as np
from bayopt.plot.stats import maximum_locus
from bayopt.plot.stats import minimum_locus
from bayopt.plot.stats import with_confidential
from bayopt.plot.stats import histogram
from bayopt.plot.stats import count_true


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
        self.assertEqual(results.values.shape[1], 8)

    def test_histogram(self):
        data = np.array([1.2, 5.5, 10])
        x, y = histogram(data=data, start=0, stop=10, step=1)

        self.assertTrue(np.all(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.str) == x))
        self.assertEqual(len(y), 10)
        self.assertTrue(np.all(np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1]) == y))

    def test_count_true(self):
        data = np.array([
            ['True', 'False', 'False'],
            ['False', 'False', 'False'],
            ['True', 'True', 'False']
        ])

        result = count_true(data)

        self.assertTrue(np.all([1, 0, 2] == result))
