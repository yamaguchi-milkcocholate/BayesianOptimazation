import unittest
from bayopt.plot.loader import load_files
from bayopt.plot.loader import load_experiments


class TestLoader(unittest.TestCase):

    def test_load_files_early(self):
        results = load_files(
            function_name='Unit Test',
            start='2019-05-01',
            end='2019-05-30',
            dim='Unit',
            fill_in_strategy='Test'
        )
        self.assertEqual(0, len(results))

    def test_load_files_late(self):
        results = load_files(
            function_name='Unit Test',
            start='2019-06-30',
            end='2019-07-30',
            dim='Unit',
            fill_in_strategy='Test'
        )
        self.assertEqual(0, len(results))

    def test_load_files_mismatch(self):
        results = load_files(
            function_name='Unit Test',
            start='2019-05-31',
            end='2019-06-30',
            dim='E',
            fill_in_strategy='E'
        )
        self.assertEqual(0, len(results))

    def test_load_files_ok(self):
        results = load_files(
            function_name='Unit Test',
            start='2019-05-31',
            end='2019-06-30',
            dim='Unit',
            fill_in_strategy='Test'
        )
        self.assertEqual(2, len(results))

    def test_load_dropout(self):
        results = load_experiments(function_name='Unit Test', dim='Unit', feature='Test')
        self.assertEqual(results.shape[0], 11)
        self.assertEqual(results.shape[1], 2)
