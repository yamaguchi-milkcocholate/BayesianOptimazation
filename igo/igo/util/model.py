#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
import sys
import numpy as np
import scipy.linalg

# public symbols
__all__ = ['Model']


class Model(object):
    """
    Base class for models
    """

    @abstractmethod
    def sampling(self, lam):
        """
        Abstract method for sampling.
        :param int lam: sample size :math:`\\lambda`
        :return: samples
        """
        pass

    @abstractmethod
    def loglikelihood(self, X):
        """
        Abstract method for log likelihood.
        :param X: samples
        :return: log likelihoods
        """
        pass

    def terminate_condition(self):
        """
        Check terminate condition.
        :return bool: terminate condition is satisfied or not
        """
        return False

    def verbose_display(self):
        """
        Return verbose display string.
        :return str: string for verbose display
        """
        return ''

    def log_header(self):
        """
        Return model log header list.
        :return: header info list for model log
        :rtype string list:
        """
        return []

    def log(self):
        """
        Return model log string list.
        :return: model log string list
        :rtype string list:
        """
        return []


class Gaussian(Model):
    """
    Gaussian distribution parametrized by mean vector :math:`m` and (full) covariance matrix :math:`C`.

    :param int d: dimension
    :param m: mean vector :math:`m` (option, default is numpy.zeros(d))
    :param C: covariance matrix :math:`C` (option, default is numpy.identity(d))
    :param float minimal_eigenval: minimal eigenvalue for terminate condition
    :type m: array_like, shape(d), dtype=float
    :type C: array_like, shape(d, d), dtype=float
    """
    def __init__(self, d, m=None, C=None, minimal_eigenval=1e-30):
        self.d = d
        self.m = m if m is not None else np.zeros(self.d)
        self.C = C if C is not None else np.identity(self.d)
        self.min_eigenval = minimal_eigenval

        if len(self.m) != d or self.C.shape != (d, d):
            print("The size of parameters is invalid.")
            print("Dimension: %d, Mean vector: %d, Covariance matrix: %s" % (self.d, len(self.m), self.__C.shape))
            print("at " + self.__class__.__name__ + " class")
            sys.exit(1)

    def _get_C(self):
        return self.__C

    def _set_C(self, C):
        self.__C = C
        self.__eigen_decomposition()

    C = property(_get_C, _set_C)

    def sampling(self, lam):
        """
        Draw :math:`\\lambda` samples from the Gaussian distribution.

        :param int lam: sample size :math:`\\lambda`
        :return: sampled vectors from :math:`\\mathcal{N}(m, C)` Gaussian distribution
        :rtype: array_like, shape=(lam, d), dtype=float
        """
        return np.random.randn(lam, self.d).dot(self.sqrtC.T) + self.m

    def loglikelihood(self, X):
        """
        Calculate log likelihood.

        :param X: samples
        :type X: array_like, shape=(lam, d), dtype=float
        :return: log likelihoods
        :rtype: array_like, shape=(lam), dtype=float
        """
        Z = np.dot((X - self.m), self.invSqrtC.T)
        return - 0.5 * (self.d * np.log(2. * np.pi) + self.logDetC) - 0.5 * np.linalg.norm(Z, axis=1)**2

    def terminate_condition(self):
        return np.min(self.eigvals) < self.min_eigenval

    def verbose_display(self):
        return ' MinEigVal: %e' % (np.min(self.eigvals))

    def log_header(self):
        return ['m%d' % i for i in range(self.d)] + ['eigval%d' % i for i in range(self.d)] + ['logDetC']

    def log(self):
        return ['%e' % i for i in self.m] + ['%e' % i for i in self.eigvals] + ['%e' % self.logDetC]

    # Private method
    def __eigen_decomposition(self):
        self.eigvals, self.eigvectors = scipy.linalg.eigh(self.C)
        B = self.eigvectors

        if np.min(self.eigvals) > 0.:
            D = np.diag(np.sqrt(self.eigvals))
            self.sqrtC = np.dot(np.dot(B, D), B.T)
            # self.invC = np.dot(np.dot(B, np.diag(np.reciprocal(self.eigvals))), B.T)
            self.invSqrtC = np.dot(np.dot(B, np.diag(np.reciprocal(np.sqrt(self.eigvals)))), B.T)
            self.logDetC = np.log(self.eigvals).sum()
        else:
            print('The minimal eigenvalue becomes negative value!')


class GaussianSigmaC(Gaussian):
    def __init__(self, d, m=None, C=None, sigma=1., minimal_eigenval=1e-30):
        super(GaussianSigmaC, self).__init__(d, m=m, C=C, minimal_eigenval=minimal_eigenval)
        self.sigma = sigma

    def sampling(self, lam):
        return self.sigma * np.random.randn(lam, self.d).dot(self.sqrtC.T) + self.m

    def loglikelihood(self, X):
        Z = np.dot((X - self.m), self.invSqrtC.T) / self.sigma
        return - 0.5 * (self.d * np.log(2. * np.pi) + self.logDetC) - np.log(self.sigma) - 0.5 * np.linalg.norm(Z, axis=1)**2

    def terminate_condition(self):
        return (self.sigma**2) * np.min(self.eigvals) < self.min_eigenval

    def verbose_display(self):
        return ' MinEigVal: %e' % ((self.sigma**2) * (np.min(self.eigvals)))

    def log_header(self):
        return super(GaussianSigmaC, self).log_header() + ['sigma']

    def log(self):
        return super(GaussianSigmaC, self).log() + ['%e' % self.sigma]


class Bernoulli(Model):
    """
    Bernoulli distribution for binary string parametrized by :math:`\\{ \\theta \\}_{i=1}^d`.

    :param int d: the number of bits
    :param theta: parameter vector for Bernoulli distribution :math:`\\theta`, the range is [0.0, 1.0] (option, default is 0.5 * numpy.ones(d))
    :type theta: array_like, shape(d), dtype=float
    """

    def __init__(self, d, theta=None):
        self.d = d
        self.theta = theta if theta is not None else 0.5 * np.ones(self.d)

        if len(self.theta) != d or (self.theta > 1).sum() > 0 or (self.theta < 0).sum() > 0:
            print("The size of parameters is invalid.")
            print("Dimension: %d, Theta: %d" % (self.d, len(self.theta)))
            print("The range of parameters is [0.0, 1.0].")
            print("theta = " + self.theta)
            print("at " + self.__class__.__name__ + " class")
            sys.exit(1)

    def sampling(self, lam):
        """
        Draw :math:`\\lambda` samples from the Bernoulli distribution.
        :param int lam: sample size :math:`\\lambda`
        :return: sampled vectors from Bernoulli distribution
        :rtype: array_like, shape=(lam, d), dtype=bool
        """
        rand = np.random.rand(lam, self.d)
        return rand < self.theta

    def loglikelihood(self, X):
        """
        Calculate log likelihood.

        :param X: samples
        :type X: array_like, shape=(lam, d), dtype=bool
        :return: log likelihoods
        :rtype: array_like, shape=(lam), dtype=float
        """
        return (X * np.log(self.theta) + (1 - X) * np.log(1. - self.theta)).sum(axis=1)

    def verbose_display(self):
        return ''
        # return '\n' + str(self.theta)

    def log_header(self):
        return ['theta%d' % i for i in range(self.d)]

    def log(self):
        return ['%f' % i for i in self.theta]
