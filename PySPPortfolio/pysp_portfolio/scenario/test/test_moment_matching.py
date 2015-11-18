# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""

from __future__ import division
from time import time
import numpy as np
import scipy.stats as spstats
import pandas as pd
from PySPPortfolio.pysp_portfolio.scenario.moment_matching import (
    heuristic_moment_matching as HMM,)

from PySPPortfolio.pysp_portfolio.scenario.c_moment_matching import (
    heuristic_moment_matching as c_HMM,)

def test_biased_HMM(precision=2):
    n_rv, n_sample = 50, 100
    n_scenario = 500
    data = np.random.rand(n_rv, n_sample)

    # original statistics
    tgt_moments = np.zeros((n_rv, 4))
    tgt_moments[:, 0] = data.mean(axis=1)
    tgt_moments[:, 1] = data.std(axis=1)
    tgt_moments[:, 2] = spstats.skew(data, axis=1)
    tgt_moments[:, 3] = spstats.kurtosis(data, axis=1)
    tgt_corrs = np.corrcoef(data)

    t0 = time()
    py_scenarios = HMM(tgt_moments, tgt_corrs, n_scenario)
    print ("python HMM (n_rv, n_scenario):({}, {}) {:.4f} secs".format(
        n_rv, n_scenario, time()-t0))

    t1 = time()
    c_scenarios = c_HMM(tgt_moments, tgt_corrs, n_scenario)
    print ("c HMM (n_rv, n_scenario):({}, {}) {:.4f} secs".format(
        n_rv, n_scenario, time()-t1))

    for scenarios in (py_scenarios, c_scenarios):
        # scenarios statistics
        res_moments = np.zeros((n_rv, 4))
        res_moments[:, 0] = scenarios.mean(axis=1)
        res_moments[:, 1] = scenarios.std(axis=1)
        res_moments[:, 2] = spstats.skew(scenarios, axis=1)
        res_moments[:, 3] = spstats.kurtosis(scenarios, axis=1)
        res_corrs = np.corrcoef(scenarios)

        np.testing.assert_array_almost_equal(tgt_moments, res_moments, precision)
        np.testing.assert_array_almost_equal(tgt_corrs, res_corrs, precision)


def test_unbiased_HMM(precision=2):
    n_rv, n_sample = 10, 100
    n_scenario = 500
    data = np.random.rand(n_rv, n_sample)

    # original statistics
    tgt_moments = np.zeros((n_rv, 4))
    tgt_moments[:, 0] = data.mean(axis=1)
    tgt_moments[:, 1] = data.std(axis=1, ddof=1)
    tgt_moments[:, 2] = spstats.skew(data, axis=1, bias=False)
    tgt_moments[:, 3] = spstats.kurtosis(data, axis=1, bias=False)
    tgt_corrs = np.corrcoef(data)

    t0 = time()
    py_scenarios = HMM(tgt_moments, tgt_corrs, n_scenario, bias=False)
    print ("python unbiased HMM (n_rv, n_scenario):({}, {}) {:.4f} secs".format(
        n_rv, n_scenario, time()-t0))

    t1 = time()
    c_scenarios = c_HMM(tgt_moments, tgt_corrs, n_scenario, bias=False)
    print ("c unbiased HMM (n_rv, n_scenario):({}, {}) {:.4f} secs".format(
        n_rv, n_scenario, time()-t1))

    for scenarios in (py_scenarios,  c_scenarios):
        # scenarios statistics
        res_moments = np.zeros((n_rv, 4))
        res_moments[:, 0] = scenarios.mean(axis=1)
        res_moments[:, 1] = scenarios.std(axis=1, ddof=1)
        res_moments[:, 2] = spstats.skew(scenarios, axis=1, bias=False)
        res_moments[:, 3] = spstats.kurtosis(scenarios, axis=1, bias=False)
        res_corrs = np.corrcoef(scenarios)

        np.testing.assert_array_almost_equal(tgt_moments, res_moments, precision)
        np.testing.assert_array_almost_equal(tgt_corrs, res_corrs, precision)

def test_moments():
    """ test equation of 1~4 th moments of scipy and padnas """
    n_rv, n_sample = 50, 100
    data = np.random.rand(n_rv, n_sample)
    pd_data = pd.DataFrame(data)

    tgt_moments = np.zeros((n_rv, 4))
    pd_moments = pd.DataFrame(np.zeros((n_rv, 4)))

    # mean
    tgt_moments[:, 0] = data.mean(axis=1)
    pd_moments.iloc[:, 0] = pd_data.mean(axis=1)
    np.testing.assert_array_almost_equal(tgt_moments[:, 0],
                                         pd_moments.iloc[:, 0])

    # std
    tgt_moments[:, 1] = data.std(axis=1, ddof=1)
    pd_moments.iloc[:, 1] = pd_data.std(axis=1)
    np.testing.assert_array_almost_equal(tgt_moments[:, 1],
                                         pd_moments.iloc[:, 1])

    # skew
    tgt_moments[:, 2] = spstats.skew(data, axis=1, bias=False)
    pd_moments.iloc[:, 2] = pd_data.skew(axis=1)
    np.testing.assert_array_almost_equal(tgt_moments[:, 2],
                                         pd_moments.iloc[:, 2])

    # kurtosis
    tgt_moments[:, 3] = spstats.kurtosis(data, axis=1, bias=False)
    pd_moments.iloc[:, 3] = pd_data.kurtosis(axis=1)
    np.testing.assert_array_almost_equal(tgt_moments[:, 3],
                                         pd_moments.iloc[:, 3])

    # correlation
    tgt_corrs = np.corrcoef(data)
    pd_corrs = (pd_data.T).corr()
    np.testing.assert_array_almost_equal(tgt_corrs, pd_corrs)


def test_skew():
    """ test equation of skew in scipy """
    n = 100
    x = np.random.rand(n)

    # biased estimator
    b_skew = spstats.skew(x, bias=True)

    s3 = sum((v-x.mean())**3 for v in x)/n
    s2 = sum((v-x.mean())**2 for v in x)/n
    b_skew2 = s3/np.power(s2, 1.5)
    print ("biased skew:", b_skew2)
    np.testing.assert_allclose(b_skew, b_skew2)

    # unbiased estimator
    ub_skew = spstats.skew(x, bias=False)

    s3 = sum((v-x.mean())**3 for v in x)/n
    s2 = sum((v-x.mean())**2 for v in x)/(n-1)
    ub_skew2 = s3/np.power(s2, 1.5) * n*n/(n-1)/(n-2)
    print ("ub biased skew:", ub_skew2)
    np.testing.assert_allclose(ub_skew, ub_skew2)

def test_kurtosis():
    """ test equation of kurtosis in scipy """
    n = 100
    x = np.random.rand(n)

    # biased estimator
    b_kurt = spstats.kurtosis(x, bias=True)

    k4 = sum((v-x.mean())**4 for v in x)/n
    k2 = sum((v-x.mean())**2 for v in x)/n
    b_kurt2 = k4/k2**2 - 3
    print ("biased kurtosis:", b_kurt2)
    np.testing.assert_allclose(b_kurt, b_kurt2)

    # unbiased estimator
    ub_kurt = spstats.kurtosis(x, bias=False)

    k4 = sum((v-x.mean())**4 for v in x)/n
    k2 = sum((v-x.mean())**2 for v in x)/n
    ub_kurt2 = 1.0/(n-2)/(n-3) * ((n**2-1.0)*k4/k2**2.0 - 3*(n-1)**2.0)
    print ("ubbiased kurtosis:", ub_kurt2)

    k2 = sum((v-x.mean())**2 for v in x)/(n-1)
    ub_kurt3 = 1/(n-2)/(n-3) *((n**2-1.0)*(n/(n-1))**2. *k4/k2**2.0- 3*(
        n-1)**2.0)
    print ("ubbiased kurtosis:", ub_kurt3)
    np.testing.assert_allclose(ub_kurt, ub_kurt2)
    np.testing.assert_allclose(ub_kurt, ub_kurt3)

if __name__ == '__main__':
    pass
    # test_biased_HMM()
    # test_unbiased_HMM()
#     # test_moments()
#     test_skew()
#     test_kurtosis()
#     test_kurtosis2()
