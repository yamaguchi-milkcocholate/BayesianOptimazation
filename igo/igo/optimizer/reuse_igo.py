#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

from ..util.model import Gaussian, Bernoulli
from ..optimizer.base_optimizer import BaseOptimizer
from ..optimizer.cmaes import CMAParam

__all__ = ['ReuseIGO', 'ReuseGaussianIGO', 'ReuseBernoulliIGO']


class ReuseIGO(BaseOptimizer):
    def __init__(self, models, d, lam, K=0):
        self.models = models
        self.d = d
        self.lam = lam
        self.K = K

        self.weights = None
        self.current_k = 0
        self.used_k_num = 0
        
        total_pop_size = (self.K + 1) * self.lam
        self.X = np.empty((total_pop_size, self.d))
        self.evals = np.empty(total_pop_size)
        self.like_ratio = np.empty(total_pop_size)
        self.loglikes = np.empty((self.K+1, total_pop_size))


    def _get_Xt(self):
        cidx = self.current_k * self.lam
        return self.X[cidx:cidx+self.lam]

    def _set_Xt(self, x):
        cidx = self.current_k * self.lam
        self.X[cidx:cidx+self.lam] = x
    Xt = property(_get_Xt, _set_Xt)

    def _get_fXt(self):
        cidx = self.current_k * self.lam
        return self.evals[cidx:cidx+self.lam]

    def _set_fXt(self, fx):
        cidx = self.current_k * self.lam
        self.evals[cidx:cidx+self.lam] = fx
    fXt = property(_get_fXt, _set_fXt)

    def all_X(self):
        return self.X[:(self.used_k_num + 1) * self.lam]

    def all_evals(self):
        return self.evals[:(self.used_k_num + 1) * self.lam]

    def all_like_ratio(self):
        return self.like_ratio[:(self.used_k_num + 1) * self.lam]

    def compute_like_ratio(self):
        cidx = self.current_k * self.lam
        psize = (self.used_k_num + 1) * self.lam
        # Compute probability density of currently generated individuals
        for k in range(self.used_k_num+1):
            self.loglikes[k, cidx:cidx+self.lam] = self.models[k].loglikelihood(self.Xt)
        # Compute probability density for current parameters
        self.loglikes[self.current_k, :psize] = self.models[self.current_k].loglikelihood(self.all_X())
        # Compute probability density ratio
        self.like_ratio[:psize] = (self.used_k_num + 1) / np.exp(self.loglikes[:self.used_k_num+1, :psize]
                                                                 - self.loglikes[self.current_k, :psize]).sum(axis=0)
        # print(self.like_ratio[:psize])
        return self.like_ratio

    def sampling_model(self):
        return self.models[self.current_k]

    def update(self, X, evals):
        pass

    def terminate_condition(self):
        return self.models[self.current_k].terminate_condition()

    def verbose_display(self):
        return self.models[self.current_k].verbose_display()

    def log_header(self):
        return self.models[self.current_k].log_header() + ['SumCoeff%d' % k for k in range(self.K + 1)]

    def log(self):
        s = np.zeros(self.K + 1)
        for k in range(self.K + 1):
            ind = (self.current_k - k - 1 + (self.K + 1)) % (self.K + 1)
            if self.weights is None or len(self.weights) <= self.lam * ind:
                s[k] = 0.
            else:
                s[k] = self.weights[self.lam * ind: self.lam * (ind + 1)].sum()
        return self.models[self.current_k].log() + ['%f' % v for v in s]


class ReuseGaussianIGO(ReuseIGO):
    def __init__(self, d, weight_func, lam, K=0, m=None, C=None, minimal_eigenval=1e-30, eta_m=1., eta_C=None):
        models = [Gaussian(d, m=m, C=C, minimal_eigenval=minimal_eigenval) for _ in range(K + 1)]
        super(ReuseGaussianIGO, self).__init__(models, d, lam, K=K)
        self.weight_func = weight_func
        self.eta_m = eta_m
        self.eta_C = eta_C if eta_C is not None else CMAParam.c_mu(d, CMAParam.mu_eff(CMAParam.pop_size(d)))

    def update(self, X, evals):
        self.Xt = X
        self.fXt = evals

        # calculate likelihood ratio
        self.compute_like_ratio()
        # weight calculation
        w = self.weight_func(self.all_evals(), likelihood_ratio=self.all_like_ratio())
        self.weights = w * self.all_like_ratio()

        # natural gradient update
        Y = self.all_X() - self.models[self.current_k].m
        WYT = self.weights * Y.T
        m_grad = WYT.sum(axis=1)
        C_grad = (np.dot(WYT, Y) - self.weights.sum() * self.models[self.current_k].C)

        next_idx = (self.current_k + 1) % (self.K + 1)
        self.models[next_idx].m = self.models[self.current_k].m + self.eta_m * m_grad
        self.models[next_idx].C = self.models[self.current_k].C + self.eta_C * C_grad

        # update current index
        if self.used_k_num < self.K:
            self.used_k_num += 1
        self.current_k = next_idx


class ReuseBernoulliIGO(ReuseIGO):
    def __init__(self, d, weight_func, lam, K=0, theta=None, eta=None, theta_max=1.0, theta_min=0.0):
        models = [Bernoulli(d, theta=theta) for _ in range(K + 1)]
        super(ReuseBernoulliIGO, self).__init__(models, d, lam, K=K)
        self.weight_func = weight_func
        self.eta = eta if eta is not None else 1. / d
        self.theta_max = theta_max
        self.theta_min = theta_min

    def update(self, X, evals):
        self.Xt = X
        self.fXt = evals

        # calculate likelihood ratio
        self.compute_like_ratio()
        # weight calculation
        w = self.weight_func(self.all_evals(), likelihood_ratio=self.all_like_ratio())
        self.weights = w * self.all_like_ratio()

        # natural gradient update
        next_idx = (self.current_k + 1) % (self.K + 1)
        self.models[next_idx].theta = self.models[self.current_k].theta + self.eta * (self.weights * (self.all_X() - self.models[self.current_k].theta).T).sum(axis=1)

        # Ensure the range of theta within [upper, lower]
        self.models[next_idx].theta = np.maximum(self.models[next_idx].theta, self.theta_min)
        self.models[next_idx].theta = np.minimum(self.models[next_idx].theta, self.theta_max)

        # update current index
        if self.used_k_num < self.K:
            self.used_k_num += 1
        self.current_k = next_idx
