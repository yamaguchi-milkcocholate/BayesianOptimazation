import unittest
from bayopt.plot.loader import load_files
from bayopt.plot.loader import load_experiments
from bayopt.plot.loader import make_uniform_by_length


class TestLoader(unittest.TestCase):

    def test_load_files_early(self):
        results = load_files(
            function_name='Unit Test',
            start='2019-05-01',
            end='2019-05-30',
            dim='Unit',
            fill_in_strategy='Test'
        )
        # self.assertEqual(0, len(results))

    def test_load_files_late(self):
        results = load_files(
            function_name='Unit Test',
            start='2019-06-30',
            end='2019-07-30',
            dim='Unit',
            fill_in_strategy='Test'
        )
        # self.assertEqual(0, len(results))

    def test_load_files_mismatch(self):
        results = load_files(
            function_name='Unit Test',
            start='2019-05-31',
            end='2019-06-30',
            dim='E',
            fill_in_strategy='E'
        )
        # self.assertEqual(0, len(results))

    def test_load_files_ok(self):
        results = load_files(
            function_name='Unit Test',
            start='2019-05-31',
            end='2019-06-30',
            dim='Unit',
            fill_in_strategy='Test'
        )
        # self.assertEqual(2, len(results))

    def test_load_dropout(self):
        results = load_experiments(function_name='Unit Test', dim='Unit', feature='Test')
        # self.assertEqual(results.shape[0], 11)
        # self.assertEqual(results.shape[1], 2)

    def test_make_uniform_by_length(self):
        data = [
            [1, 2, 3],
            [1, 2],
            [3, 4, 1, 2],
        ]
        result = make_uniform_by_length(data)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [1, 2])
        self.assertEqual(result[1], [1, 2])
        self.assertEqual(result[2], [3, 4])
