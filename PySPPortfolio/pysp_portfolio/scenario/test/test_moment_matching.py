# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from time import time
import numpy as np
import scipy.stats as spstats
from PySPPortfolio.pysp_portfolio.scenario.moment_matching import (
    heuristic_moment_matching as HMM, central_to_orig_moment)

from PySPPortfolio.pysp_portfolio.scenario.c_moment_matching import (
    heuristic_moment_matching as c_HMM,)

def test_HMM(precision=2):
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
    print "python HMM (n_rv, n_scenario):({}, {}) {:.4f} secs".format(
        n_rv, n_scenario, time()-t0)

    t1 = time()
    c_scenarios = c_HMM(tgt_moments, tgt_corrs, n_scenario)
    print "c HMM (n_rv, n_scenario):({}, {}) {:.4f} secs".format(
        n_rv, n_scenario, time()-t1)

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
#
# def test_rmse():
#     n_row, n_col = 100, 2
#     arr1 = np.random.rand(n_row, n_col)
#     arr2 = np.random.rand(n_row, n_col)
#     error = np.sqrt(((arr1 -arr2) ** 2).sum())
#
#     np.testing.assert_approx_equal(error, c_rmse(arr1, arr2))
#
# def test_central_to_orig_moment():
#     n_row, n_col = 100, 4
#     arr = np.random.rand(n_row, n_col)
#     np.testing.assert_array_almost_equal(central_to_orig_moment(arr),
#                                          c_central_to_orig_moment(arr))
#


if __name__ == '__main__':
    test_HMM()
    # test_rmse()
    # test_central_to_orig_moment()
    # c_HMM(np.random.rand(2,4), np.random.rand(2,2))

