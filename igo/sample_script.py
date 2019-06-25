#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import igo.objective_function.binary as f_bin
import igo.objective_function.continuous as f_cont
import igo.optimizer.cmaes as cma
import igo.optimizer.igo as igo
import igo.optimizer.reuse_igo as reuse_igo
import igo.optimizer.reuse_cma as reuse_cma
import igo.util.sampler as sampler
import igo.util.weight as weight


def gaussian_igo():
    # problem definition
    d = 20
    f = f_cont.Ellipsoid(d, target_eval=1e-10, max_eval=1e6)
    init_m, init_sigma = f_cont.initial_setting_for_gaussian(f)
    # parameter setting
    lam = cma.CMAParam.pop_size(d)
    # weight function
    w_func = weight.CMAWeight(lam)
    # learning rate
    eta_c = cma.CMAParam.c_mu(d, cma.CMAParam.mu_eff(lam))
    # optimizer
    opt = igo.GaussianIGO(d, w_func, m=init_m, C=(init_sigma**2) * np.identity(d), eta_m=1.0, eta_C=eta_c)
    # run
    return opt.run(sampler.DefaultSampler(f, lam), logger=None, verbose=True)
    #return opt.run(sampler.ImportanceMixingSampler(f, lam, rate=0.01), logger=None, verbose=True)


def gaussian_reuse_igo():
    # problem definition
    d = 20
    f = f_cont.Rosenbrock(d, target_eval=1e-10, max_eval=1e6)
    init_m, init_sigma = f_cont.initial_setting_for_gaussian(f)
    # parameter setting
    lam = cma.CMAParam.pop_size(d)
    K = 5
    # weight function
    non_inc_f = weight.CMANonIncFunc()
    w_func = weight.QuantileBasedWeight(non_inc_f, tie_case=True, normalization=False, min_problem=True)
    # learning rate
    eta_c = cma.CMAParam.c_mu(d, cma.CMAParam.mu_eff(lam))
    # optimizer
    opt = reuse_igo.ReuseGaussianIGO(d, w_func, lam, K=K, m=init_m, C=(init_sigma**2) * np.identity(d), eta_m=1., eta_C=eta_c)
    # run
    return opt.run(sampler.DefaultSampler(f, lam), logger=None, verbose=True)


def cma_run():
    # problem definition
    d = 20
    f = f_cont.Rosenbrock(d, target_eval=1e-10, max_eval=1e6)
    init_m, init_sigma = f_cont.initial_setting_for_gaussian(f)
    lam = cma.CMAParam.pop_size(d)
    # weight function
    w_func = weight.CMAWeight(lam)
    # optimizer
    opt = cma.CMAES(d, w_func, m=init_m, sigma=init_sigma)
    # run
    return opt.run(sampler.DefaultSampler(f, lam), logger=None, verbose=True)


def reuse_cma_run():
    # problem definition
    d = 20
    f = f_cont.Rosenbrock(d, target_eval=1e-10, max_eval=1e6)
    init_m, init_sigma = f_cont.initial_setting_for_gaussian(f)
    # parameter setting
    lam = cma.CMAParam.pop_size(d)
    K = 5
    # weight function
    non_inc_f = weight.CMANonIncFunc()
    w_func = weight.QuantileBasedWeight(non_inc_f, tie_case=True, normalization=False, min_problem=True)
    # optimizer
    opt = reuse_cma.ReuseCMA(d, w_func, lam, K=K, m=init_m, sigma=init_sigma, csa=True, rank1=True, reuse_m=False)
    # run
    return opt.run(sampler.DefaultSampler(f, lam), logger=None, verbose=True)


def bernoulli_igo():
    # problem definition
    d = 32
    f = f_bin.OneMax(d, max_eval=d**2)
    # f = f_bin.LeadingOne(dim, max_eval=dim**3)
    
    # parameter setting
    lam = 2
    theta_min = 1./d
    theta_max = 1. - 1./d
    # weight function
    non_inc_f = weight.SelectionNonIncFunc(threshold=0.25, negative_weight=True)
    w = weight.QuantileBasedWeight(non_inc_f=non_inc_f, tie_case=True, normalization=False, min_problem=False)
    # learning rate
    eta = 1./d
    # model
    opt = igo.BernoulliIGO(d, w, eta=eta, theta_max=theta_max, theta_min=theta_min)
    # run
    return opt.run(sampler.DefaultSampler(f, lam), logger=None, verbose=True)


def bernoulli_reuse_igo():
    # problem definition
    d = 32
    f = f_bin.OneMax(d, max_eval=d**2)
    # f = f_bin.LeadingOne(dim, max_eval=dim**3)
    
    # parameter setting
    lam = 2
    K = 5
    theta_min = 1./d
    theta_max = 1. - 1./d
    # weight function
    non_inc_f = weight.SelectionNonIncFunc(threshold=0.25, negative_weight=True)
    w = weight.QuantileBasedWeight(non_inc_f=non_inc_f, tie_case=True, normalization=False, min_problem=False)
    # learning rate
    eta = 1./d
    # model
    opt = reuse_igo.ReuseBernoulliIGO(d, w, lam, K=K, eta=eta, theta_max=theta_max, theta_min=theta_min)
    # run
    return opt.run(sampler.DefaultSampler(f, lam), logger=None, verbose=True)


if __name__ == '__main__':
    print(gaussian_igo())
    # print(gaussian_reuse_igo())
    # print(reuse_cma_run())
    # print(bernoulli_igo())
    # print(bernoulli_reuse_igo())
