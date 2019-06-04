import unittest
import numpy as np
from bayopt.methods.dropout import Dropout
from tests.utils.example_function import ExampleFunction
from GPyOpt.models.gpmodel import *
from GPyOpt.models.warpedgpmodel import *
from GPyOpt.acquisitions.EI import *
from GPyOpt.acquisitions.EI_mcmc import *
from GPyOpt.acquisitions.MPI import *
from GPyOpt.acquisitions.MPI_mcmc import *
from GPyOpt.acquisitions.LCB import *
from GPyOpt.acquisitions.LCB_mcmc import *


class TestDropout(unittest.TestCase):

    def setUp(self) -> None:
        self.f = ExampleFunction()

        self.domain = [
            {'name': 'x0', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
            {'name': 'x1', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
            {'name': 'x2', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
            {'name': 'x3', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
            {'name': 'x4', 'type': 'continuous', 'domain': (-3, 3), 'dimensionality': 1},
        ]

    def test_init(self):
        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random'
        )

    def test_acquisition_type(self):
        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random',
            acquisition_type='EI'
        )
        self.assertTrue(isinstance(method.acquisition, AcquisitionEI))

        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random',
            acquisition_type='MPI'
        )
        self.assertTrue(isinstance(method.acquisition, AcquisitionMPI))

        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random',
            acquisition_type='LCB'
        )
        self.assertTrue(isinstance(method.acquisition, AcquisitionLCB))

    def test_acquisition_optimizer_type(self):
        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random',
            acquisition_optimizer_type='lbfgs'
        )
        self.assertEqual(method.acquisition_optimizer_type, 'lbfgs')

        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random',
            acquisition_optimizer_type='DIRECT'
        )
        self.assertEqual(method.acquisition_optimizer_type, 'DIRECT')

        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random',
            acquisition_optimizer_type='CMA'
        )
        self.assertEqual(method.acquisition_optimizer_type, 'CMA')

    def test_model_type(self):
        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random',
            model_type='GP'
        )
        self.assertTrue(isinstance(method.model, GPModel))

        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random',
            model_type='sparseGP'
        )
        self.assertTrue(isinstance(method.model, GPModel))

        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random',
            model_type='GP_MCMC'
        )
        self.assertTrue(isinstance(method.model, GPModel_MCMC))

        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random',
            model_type='warpedGP'
        )
        self.assertTrue(isinstance(method.model, WarpedGPModel))

    def test_domain(self):
        method = Dropout(
            f=self.f, domain=self.domain, subspace_dim_size=3, fill_in_strategy='random'
        )
        self.assertEqual(method.domain[0]['name'], '0')
        self.assertEqual(method.domain[1]['name'], '1')
        self.assertEqual(method.domain[2]['name'], '2')
        self.assertEqual(method.domain[3]['name'], '3')
        self.assertEqual(method.domain[4]['name'], '4')


if __name__ == '__main__':
    unittest.main()
