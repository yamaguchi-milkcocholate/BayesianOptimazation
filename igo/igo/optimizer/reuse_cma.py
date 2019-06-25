#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import scipy.linalg

from ..util.weight import CMAWeight
from ..optimizer.cmaes import CMAParam
from ..optimizer.reuse_igo import ReuseIGO
from ..util.model import GaussianSigmaC

__all__ = ['ReuseCMA']


class ReuseCMA(ReuseIGO):
    def __init__(self, d, weight_func, lam, K=0, m=None, C=None, sigma=1., minimal_eigenval=1e-30, c_mu=None,
                 csa=False, rank1=False, reuse_m=True, path_mmove=False, recalc_mueff=False, alpha_mu=2.):
        models = [GaussianSigmaC(d, m=m, C=C, sigma=sigma, minimal_eigenval=minimal_eigenval) for _ in range(K + 1)]
        super(ReuseCMA, self).__init__(models, d, lam, K=K)
        self.weight_func = weight_func

        # Algorithm components
        self.csa = csa
        self.rank1 = rank1
        self.reuse_m = reuse_m
        self.path_mmove = path_mmove
        self.recalc_mueff = recalc_mueff

        # CMA parameters
        self.alpha_mu = alpha_mu
        self.mu_eff = CMAParam.mu_eff(self.lam)
        self.c_1 = CMAParam.c_1(d, self.mu_eff) if self.rank1 else 0.
        self.c_mu = CMAParam.c_mu(d, self.mu_eff, c1=self.c_1, alpha_mu=self.alpha_mu) if c_mu is None else c_mu
        self.c_c = CMAParam.c_c(d, self.mu_eff)
        self.c_sigma = CMAParam.c_sigma(d, self.mu_eff)
        self.damping = CMAParam.damping(d, self.mu_eff) if self.csa else np.inf
        self.chi_d = CMAParam.chi_d(d)

        # evolution path
        self.ps = np.zeros(d)
        self.pc = np.zeros(d)
        self.gen_count = 0

        # CMA weights
        self.cma_weights = CMAWeight(lam)
        self.weights = None

    # re-calculate CMA parameters
    def ReCalcMueff(self):
        self.mu_eff = CMAParam.mu_eff(self.lam, weights=self.weights)
        self.c_1 = CMAParam.c_1(self.d, self.mu_eff) if self.rank1 else 0.
        self.c_mu = CMAParam.c_mu(self.d, self.mu_eff, c1=self.c_1, alpha_mu=self.alpha_mu)
        self.c_c = CMAParam.c_c(self.d, self.mu_eff)
        self.c_sigma = CMAParam.c_sigma(self.d, self.mu_eff)
        self.damping = CMAParam.damping(self.d, self.mu_eff) if self.csa else np.inf
        self.chi_d = CMAParam.chi_d(self.d)

    def update(self, X, evals):
        self.gen_count += 1
        self.Xt = X
        self.fXt = evals

        # calculate likelihood ratio
        self.compute_like_ratio()
        # weight calculation
        w = self.weight_func(self.all_evals(), likelihood_ratio=self.all_like_ratio())
        psize = (self.used_k_num + 1) * self.lam
        self.weights = w * self.like_ratio[:psize]

        # re-calculate CMA parameters
        if self.recalc_mueff is True:
            self.ReCalcMueff()

        # current model
        c_model = self.models[self.current_k]

        # CMA weights
        cma_w = self.cma_weights(evals)
        m_grad_cma = np.sum(cma_w * (X - c_model.m).T, axis=1)

        # natural gradient
        Y = (self.all_X() - c_model.m) / c_model.sigma
        WYT = self.weights * Y.T
        m_grad = c_model.sigma * WYT.sum(axis=1) if self.reuse_m else m_grad_cma
        C_grad = np.dot(WYT, Y) - self.weights.sum() * c_model.C

        hsig = 1.
        if self.rank1 or self.csa:
            pm = m_grad if self.path_mmove else m_grad_cma
            # evolution path
            self.ps = (1.0 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * np.dot(c_model.invSqrtC, pm / c_model.sigma)
            hsig = 1. if scipy.linalg.norm(self.ps) / (np.sqrt(1. - (1. - self.c_sigma) ** (2 * self.gen_count))) < (1.4 + 2. / (c_model.d + 1.)) * self.chi_d else 0.
            self.pc = (1. - self.c_c) * self.pc + hsig * np.sqrt(self.c_c * (2. - self.c_c) * self.mu_eff) * pm / c_model.sigma

        next_idx = (self.current_k + 1) % (self.K + 1)

        if self.csa:
            # CSA
            self.models[next_idx].sigma = c_model.sigma * np.exp(self.c_sigma / self.damping * (scipy.linalg.norm(self.ps) / self.chi_d - 1.))

        # mean vector update
        self.models[next_idx].m = c_model.m + m_grad
        # covariance matrix update
        self.models[next_idx].C = c_model.C + (1. - hsig) * self.c_1 * self.c_c * (2. - self.c_c) * c_model.C + self.c_1 * (np.outer(self.pc, self.pc) - c_model.C) + self.c_mu * C_grad

        # update current index
        if self.used_k_num < self.K:
            self.used_k_num += 1
        self.current_k = next_idx

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
