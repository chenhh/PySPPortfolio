# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2
"""
from time import time
import numpy as np
import scipy.stats as spstats
from PySPPortfolio.pysp_portfolio.scenario.moment_matching import (
    heuristic_moment_matching,)

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
    scenarios = heuristic_moment_matching(tgt_moments, tgt_corrs, n_scenario)
    print "python HMM (n_rv, n_scenario):({}, {}) {:.4f} secs".format(
        n_rv, n_scenario, time()-t0)

    # scenarios statistics
    res_moments = np.zeros((n_rv, 4))
    res_moments[:, 0] = scenarios.mean(axis=1)
    res_moments[:, 1] = scenarios.std(axis=1)
    res_moments[:, 2] = spstats.skew(scenarios, axis=1)
    res_moments[:, 3] = spstats.kurtosis(scenarios, axis=1)
    res_corrs = np.corrcoef(scenarios)

    np.testing.assert_array_almost_equal(tgt_moments, res_moments, precision)
    np.testing.assert_array_almost_equal(tgt_corrs, res_corrs, precision)

if __name__ == '__main__':
    test_HMM()


