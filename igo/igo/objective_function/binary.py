#!/usr/bin/env python
# -*- coding: utf-8 -*-


from ..objective_function.base import *
import numpy as np


# public symbols
__all__ = ['OneMax', 'LeadingOne', 'kTrap', 'RoyalRoad1']


class OneMax(ObjectiveFunction):
    minimization_problem = False

    def __init__(self, d, target_eval=None, max_eval=1e4):
        self.d = d
        if target_eval is None:
            target_eval = self.d
        super(OneMax, self).__init__(target_eval, max_eval)

    def __call__(self, X):
        """
        Evaluation.

        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=bool
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        evals = X.sum(axis=1)
        self._update_best_eval(evals)
        return evals


class LeadingOne(ObjectiveFunction):
    minimization_problem = False

    def __init__(self, d, target_eval=None, max_eval=1e4):
        self.d = d
        if target_eval is None:
            target_eval = self.d
        super(LeadingOne, self).__init__(target_eval, max_eval)

    def __call__(self, X):
        """
        Evaluation.

        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=bool
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        evals = X.argmin(axis=1) + X.prod(axis=1) * self.d
        self._update_best_eval(evals)
        return evals


class kTrap(ObjectiveFunction):
    minimization_problem = False

    def __init__(self, block_num, k=3, target_eval=None, max_eval=1e4):
        self.d = k * block_num
        self.k = k
        self.block_num = block_num
        if target_eval is None:
            target_eval = self.d
        super(kTrap, self).__init__(target_eval, max_eval)

    def block_fit(self, block):
        fit = block.sum(axis=1)
        fit[fit != self.k] = self.k - 1 - fit[fit != self.k]
        return fit

    def __call__(self, X):
        """
        Evaluation.

        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=bool
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        evals = np.zeros(len(X))
        for i in range(0, self.d, self.k):
            evals += self.block_fit(X[:, i:i+self.k])
        self._update_best_eval(evals)
        return evals


class RoyalRoad1(ObjectiveFunction):
    minimization_problem = False

    def __init__(self, target_eval=None, max_eval=1e4):
        self.d = 64
        if target_eval is None:
            target_eval = self.d
        super(RoyalRoad1, self).__init__(target_eval, max_eval)

    def __call__(self, X):
        """
        Evaluation.

        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=bool
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        evals = np.zeros(len(X))
        s1 = X[:,  0: 8].sum(axis=1)
        evals[s1 == 8] += 8
        s2 = X[:,  8:16].sum(axis=1)
        evals[s2 == 8] += 8
        s3 = X[:, 16:24].sum(axis=1)
        evals[s3 == 8] += 8
        s4 = X[:, 24:32].sum(axis=1)
        evals[s4 == 8] += 8
        s5 = X[:, 32:40].sum(axis=1)
        evals[s5 == 8] += 8
        s6 = X[:, 40:48].sum(axis=1)
        evals[s6 == 8] += 8
        s7 = X[:, 48:56].sum(axis=1)
        evals[s7 == 8] += 8
        s8 = X[:, 56:64].sum(axis=1)
        evals[s8 == 8] += 8
        self._update_best_eval(evals)
        return evals
