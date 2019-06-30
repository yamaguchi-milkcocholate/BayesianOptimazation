#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np

from ..util.model import Gaussian, Bernoulli
from ..optimizer.base_optimizer import BaseOptimizer
from ..optimizer.cmaes import CMAParam

# public symbols
__all__ = ['GaussianIGO', 'BernoulliIGO']


class GaussianIGO(BaseOptimizer):
    """
    Gaussian IGO parametrized by mean vector :math:`m` and (full) covariance matrix :math:`C`.
    """

    def __init__(self, d, weight_func, m=None, C=None, minimal_eigenval=1e-30, eta_m=1.0, eta_C=None):
        self.model = Gaussian(d, m=m, C=C, minimal_eigenval=minimal_eigenval)
        self.weight_func = weight_func
        self.eta_m = eta_m
        self.eta_C = eta_C if eta_C is not None else CMAParam.c_mu(d, CMAParam.mu_eff(CMAParam.pop_size(d)))

    def sampling_model(self):
        return self.model

    def update(self, X, evals):
        weights = self.weight_func(evals)
        Y = X - self.model.m
        WYT = weights * Y.T
        m_grad = WYT.sum(axis=1)
        C_grad = (np.dot(WYT, Y) - weights.sum() * self.model.C)

        # natural gradient update
        self.model.m = self.model.m + self.eta_m * m_grad
        self.model.C = self.model.C + self.eta_C * C_grad

    def terminate_condition(self):
        return self.model.terminate_condition()

    def verbose_display(self):
        return self.model.verbose_display()

    def log_header(self):
        return self.model.log_header()

    def log(self):
        return self.model.log()


class BernoulliIGO(BaseOptimizer):
    """
    Bernoulli distribution for binary string parametrized by :math:`\\{ \\theta \\}_{i=1}^d`.

    :param int d: the number of bits
    :param theta: parameter vector for Bernoulli distribution :math:`\\theta`, the range is [0.0, 1.0] (option, default is 0.5 * numpy.ones(d))
    :type theta: array_like, shape(d), dtype=float
    :param float eta: learning rate
    """

    def __init__(self, d, weight_func, theta=None, eta=None, theta_max=1.0, theta_min=0.0):
        self.model = Bernoulli(d, theta=theta)
        self.weight_func = weight_func
        self.eta = eta if eta is not None else 1./d
        self.theta_max = theta_max
        self.theta_min = theta_min

        if len(self.model.theta) != d:
            print("The size of parameters is invalid.")
            print("Dimension: %d, Theta: %d" % (self.model.d, len(self.model.theta)))
            print("at " + self.__class__.__name__ + " class")
            sys.exit(1)

    def sampling_model(self):
        return self.model

    def update(self, X, evals):
        """
        Natural gradient update (:math:`\\theta = \\theta + \\eta \\sum_i^{\\lambda} w_i \\tilde{\\nabla} \\log p(x_i|\\theta)`).
        """
        weights = self.weight_func(evals)
        self.model.theta = self.model.theta + self.eta * (weights * (X - self.model.theta).T).sum(axis=1)

        # Ensure the range of theta within [upper, lower]
        self.model.theta = np.maximum(self.model.theta, self.theta_min)
        self.model.theta = np.minimum(self.model.theta, self.theta_max)

    def terminate_condition(self):
        return self.model.terminate_condition()

    def verbose_display(self):
        return self.model.verbose_display()

    def log_header(self):
        return self.model.log_header()

    def log(self):
        return self.model.log()
