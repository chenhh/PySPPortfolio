# -*- coding: utf-8 -*-
"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2

Høyland, K.; Kaut, M. & Wallace, S. W., "A heuristic for
moment-matching scenario generation," Computational optimization
and applications, vol. 24, pp 169-185, 2003.

"""

from __future__ import division
import os
from datetime import date
import pandas as pd
import numpy as np
import numpy.linalg as la
import scipy.optimize as spopt
import scipy.stats as spstats
import matplotlib.pyplot as plt
from time import time


def heuristic_moment_matching(tgt_moments, tgt_corrs,
                              n_scenario=200, n_data=0, bias=True,
                              max_moment_err=1e-3, max_corr_err=1e-3,
                              max_cubic_err=1e-5, verbose=False):
    """
    Parameters:
    --------------
    tgt_moments:, numpy.array,shape: (n_rv * 4), 1~4 central moments
    tgt_corrs:, numpy.array, size: shape: (n_rv * n_rv), correlation matrix
    n_scenario:, positive integer, number of scenario to generate
    n_data: positive integer, number of data (window length) for estimating
        moments, it is required while bias is False.
    bias: boolean,
        - True means population estimators,
        - False means sample estimators
    max_err_moment: float, max moment of error between tgt_moments and
        sample moments
    max_err_corr: float, max moment of error between tgt_corrs and
        sample correlation matrix

    Returns:
    -------------
    out_mtx: numpy.array, shape:(n_rv, n_scenario)
    """

    # check variable
    assert n_scenario >= 0
    assert tgt_moments.shape[0] == tgt_corrs.shape[0] == tgt_corrs.shape[1]
    t0 = time()

    if not bias and n_data <= 0:
        raise ValueError('n_data should be positive value')

    # parameters
    n_rv = tgt_moments.shape[0]

    # iteration for find good start samples
    max_start_iter = 5

    # cubic transform iteration
    max_cubic_iter = 2

    # main iteration of moment matching loop
    max_main_iter = 20

    # out mtx, for storing scenarios
    out_mtx = np.empty((n_rv, n_scenario))

    # ***********************************************************
    # sample target moments Y shape: (n_rv, 4) with
    # zero mean, and unit variance,
    # the same 3rd, 4th moments as target moments.
    # After generating samples X, we can transform W = aX+b
    # where the 1~4 moments of W are the same as tgt_moments
    # ***********************************************************
    y_moments = np.zeros((n_rv, 4))
    y_moments[:, 1] = 1
    y_moments[:, 2] = tgt_moments[:, 2]
    if bias:
        y_moments[:, 3] = tgt_moments[:, 3] + 3
    else:
        y_moments[:, 3] = (tgt_moments[:, 3] +
                           3.*(n_data-1)**2/(n_data-2)/(n_data-3))

    # find good start moment matrix (with err_moment converge)
    for rv in xrange(n_rv):
        cubic_err, best_cub_err = float('inf'), float('inf')

        # loop until errMom converge
        for _ in xrange(max_start_iter):
            # each random variable consists of n_scenario random sample
            tmp_out = np.random.rand(n_scenario)

            # 1~4th moments of the random variable, shape (4, )
            ey = y_moments[rv, :]

            # loop until cubic transform converge
            for cub_iter in xrange(max_cubic_iter):

                # 1~12th moments of the random samples
                ex = np.array([(tmp_out ** (idx + 1)).mean()
                                  for idx in xrange(12)])

                # find corresponding cubic parameters
                x_init = np.array([0., 1., 0., 0.])
                out = spopt.leastsq(cubic_function, x_init, args=(ex, ey),
                                    full_output=True, ftol=1E-12,
                                    xtol=1E-12)
                cubic_params = out[0]
                cubic_err = np.sum(out[2]['fvec'] ** 2)

                # update random samples
                tmp_out = (cubic_params[0] +
                           cubic_params[1] * tmp_out +
                           cubic_params[2] * (tmp_out ** 2) +
                           cubic_params[3] * (tmp_out ** 3))

                if cubic_err < max_cubic_err:
                    # find good samples
                    break
                else:
                    if verbose:
                        print ("rv:{}, cubiter:{}, cubErr: {}, "
                              "not converge".format(rv, cub_iter, cubic_err))

            # accept current samples
            if cubic_err < best_cub_err:
                best_cub_err = cubic_err
                out_mtx[rv, :] = tmp_out

    # computing starting properties and error
    # correct moment, wrong correlation

    moments_err, corrs_err = error_statistics(out_mtx, y_moments,
                                              tgt_corrs)
    if verbose:
        print ('start mtx (orig) moment_err:{}, corr_err:{}'.format(
            moments_err, corrs_err))

    # Cholesky decomposition of target corr mtx
    c_lower = la.cholesky(tgt_corrs)

    # main iteration, break when converge
    for main_iter in xrange(max_main_iter):
        if moments_err < max_moment_err and corrs_err < max_corr_err:
            break

        # transform matrix
        out_corrs = np.corrcoef(out_mtx)
        co_inv = la.inv(la.cholesky(out_corrs))
        l_vec = np.dot(c_lower, co_inv)
        out_mtx = np.dot(l_vec, out_mtx)

        # wrong moment, but correct correlation
        moments_err, corrs_err = error_statistics(out_mtx, y_moments,
                                                  tgt_corrs)
        if verbose:
            print ('main_iter:{} cholesky transform (orig) moment_err:{}, '
                  'corr_err:{}'.format(main_iter, moments_err, corrs_err))

        # after Cholesky decompsition ,the corr_err converges,
        # but the moment error may enlarge, hence it requires
        # cubic transform again
        for rv in xrange(n_rv):
            cubic_err = float('inf')
            tmp_out = out_mtx[rv, :]
            ey = y_moments[rv, :]

            # loop until cubic transform error converge
            for cub_iter in xrange(max_cubic_iter):
                ex = np.array([(tmp_out ** (idx + 1)).mean()
                                  for idx in xrange(12)])
                X_init = np.array([0., 1., 0., 0.])
                out = spopt.leastsq(cubic_function, X_init, args=(ex, ey),
                                    full_output=True, ftol=1E-12, xtol=1E-12)
                cubic_params = out[0]
                cubic_err = np.sum(out[2]['fvec'] ** 2)

                tmp_out = (cubic_params[0] +
                           cubic_params[1] * tmp_out +
                           cubic_params[2] * (tmp_out ** 2) +
                           cubic_params[3] * (tmp_out ** 3))

                if cubic_err < max_cubic_err:
                    out_mtx[rv, :] = tmp_out
                    break
                else:
                    if verbose:
                        print ("main_iter:{}, rv: {}, "
                              "(orig) cub_iter:{}, "
                              "cubErr: {}, not converge".format(
                            main_iter, rv, cub_iter, cubic_err))

        moments_err, corrs_err = error_statistics(out_mtx, y_moments,
                                                  tgt_corrs)
        if verbose:
            print ('main_iter:{} cubic_transform, (orig) moment eror:{}, '
                  'corr err: {}'.format(main_iter, moments_err, corrs_err))

    # rescale data to original moments, out_mtx shape:(n_rv, n_scenario)
    out_mtx = (out_mtx * tgt_moments[:, 1][:, np.newaxis] +
               tgt_moments[:, 0][:, np.newaxis])

    out_central_moments = np.empty((n_rv, 4))
    out_central_moments[:, 0] = out_mtx.mean(axis=1)
    if bias:
        out_central_moments[:, 1] = out_mtx.std(axis=1)
        out_central_moments[:, 2] = spstats.skew(out_mtx, axis=1)
        out_central_moments[:, 3] = spstats.kurtosis(out_mtx, axis=1)
    else:
        out_central_moments[:, 1] = out_mtx.std(axis=1, ddof=1)
        out_central_moments[:, 2] = spstats.skew(out_mtx, axis=1, bias=False)
        out_central_moments[:, 3] = spstats.kurtosis(out_mtx, axis=1,bias=False)
    out_corrs = np.corrcoef(out_mtx)

    if verbose:
        print ("1st moments difference {}".format(
            (tgt_moments[:, 0] - out_central_moments[:, 0]).sum()))
        print ("2nd moments difference {}".format(
            (tgt_moments[:, 1] - out_central_moments[:, 1]).sum()))
        print ("3th moments difference {}".format(
            (tgt_moments[:, 2] - out_central_moments[:, 2]).sum()))
        print ("4th moments difference {}".format(
            (tgt_moments[:, 3] - out_central_moments[:, 3]).sum()))
        print ("corr difference {}".format(
            (tgt_corrs - np.corrcoef(out_mtx)).sum()))

    moments_err = rmse(out_central_moments, tgt_moments)
    corrs_err = rmse(out_corrs, tgt_corrs)
    if verbose:
        print ('sample central moment err:{}, corr err:{}'.format(
            moments_err, corrs_err))
        print ("HeuristicMomentMatching elapsed {:.3f} secs".format(
            time() - t0))

    if moments_err > max_moment_err or corrs_err > max_corr_err:
        raise ValueError("out mtx not converge, moment error: {}, "
                         "corr err:{}".format(moments_err, corrs_err))

    return out_mtx


def cubic_function(cubic_params, sample_moments, tgt_moments):
    """
    Parameters:
    ----------------
    cubic_params: (a,b,c,d), four floats
    sample_moments: numpy.array, shape:(12,), 1~12 moments of samples
    tgt_moments: numpy.array, shape:(4,), 1~4th moments of target
    """
    a, b, c, d = cubic_params
    ex = sample_moments
    ey = tgt_moments

    a2 = a*a
    a3 = a2*a
    a4 = a2*a2

    b2 = b*b
    b3 = b2*b
    b4 = b2*b2

    c2 = c*c
    c3 = c2 * c
    c4 = c2*c2

    d2 = d*d
    d3 = d2*d
    d4 = d2*d2

    ab = a*b
    ac = a*c
    ad = a*d
    bd = b*d
    bc = b*c
    bcd = bc*d
    cd = c*d

    v1 = (a + b * ex[0] + c * ex[1] + d * ex[2] - ey[0])

    v2 = (d2 * ex[5] +
          2 * c * d * ex[4] +
          (2 * bd + c2) * ex[3] +
          (2 * ad + 2 * bc) * ex[2] +
          (2 * ac + b2) * ex[1] +
          2 * ab * ex[0] +
          a2 - ey[1])

    v3 = ((d3) * ex[8] +
          (3 * c * d2) * ex[7] +
          (3 * b * d2 + 3 * c2 * d) * ex[6] +
          (3 * a * d2 + 6 * bcd + c3) * ex[5] +
          (6 * ac * d + 3 * b2 * d + 3 * b * c2) * ex[4] +
          (a * (6 * bd + 3 * c2) + 3 * b2 * c) * ex[3] +
          (3 * a2 * d + 6 * a * bc + b3) * ex[2] +
          (3 * a2 * c + 3 * a * b2) * ex[1] +
           3 * a2 * b * ex[0] +
           a3 - ey[2])

    v4 = (d4 * ex[11] +
          (4 * cd * d2) * ex[10] +
          (4 * bd * d2 + 6 * c2 * d2) * ex[9] +
          4 * (ad * d2+ 3 * bc * d2 + c3 * d) * ex[8] +
          (12 * ac * d2 + 6 * b2 * d2 + 12 * bd * c2 + c4) * ex[7] +
          4 * (3 * ad * (bd + c2) + bc * (3 * bd + c2)) * ex[6] +
          (6 * a2 * d2 + ac * (24 * bd + 4 * c2) +
           4 * b3 * d + 6 * b2 * c2) * ex[5] +
          (12 * a2 * cd +  12 * ab * (bd + c2) + 4 * b2 * bc) * ex[4] +
          (a2 * (12 * bd + 6 * c2) + 12 * ac * b2  + b4) * ex[3] +
          4 * a * (a * ad + 3 * a * bc + b3) * ex[2] +
          a2 * (4 * ac + 6 * b2) * ex[1] +
          (4 * a2 * ab) * ex[0] +
          a4 - ey[3])

    return v1, v2, v3, v4


def error_statistics(out_mtx, tgt_moments, tgt_corrs=None):
    """
    Parameters:
    ----------------
    out_mtx: numpy.array, shape: (n_rv, n_scenario)
    tgt_moments: numpy.array, shape: (n_rv, 4)
    tgt_corrs: numpy.array, shape: (n_rv, n_rv)
    """
    n_rv = out_mtx.shape[0]
    out_moments = np.empty((n_rv, 4))

    for idx in xrange(4):
        out_moments[:, idx] = (out_mtx ** (idx + 1)).mean(axis=1)

    moments_err = rmse(out_moments, tgt_moments)

    if tgt_corrs is None:
        return moments_err
    else:
        out_corrs = np.corrcoef(out_mtx)
        corrs_err = rmse(out_corrs, tgt_corrs)
        return moments_err, corrs_err


def rmse(src_arr, tgt_arr):
    """
    :src_arr:, numpy.array
    :tgt_arr:, numpy.array
    """
    assert src_arr.shape == tgt_arr.shape
    error = np.sqrt(((src_arr - tgt_arr) ** 2).sum())
    return error


def central_to_orig_moment(central_moments, n_data=0, bias=True):
    """
    central moments to original moments

    for bias estimators:
        E[X] = samples.mean()
        std**2 = var = E[X**2] - E[X]*E[X]

        scipy.stats.skew, scipy.stats.kurtosis bias equations:
        m2 = np.mean((d - d.mean())**2)
        m3 = np.mean((d - d.mean())**3)
        m4 = np.mean((d - d.mean())**4)
        skew =  m3/np.sqrt(m2)**3
        kurt = m4/m2**2 -3

    for unbiased estimators:
    """
    n_rv = central_moments.shape[0]
    m1, m2, m3, m4 = (central_moments[:, 0], central_moments[:, 1],
                      central_moments[:, 2], central_moments[:, 3])
    orig_moments = np.empty((n_rv, 4))
    orig_moments[:, 0] = m1

    if bias:
        orig_moments[:, 1] = (m2 ** 2 + m1 ** 2)
        orig_moments[:, 2] = (m3 * m2 ** 3 +
                              m1 ** 3 +
                              3 *m1 * m2 ** 2)
        orig_moments[:, 3] = ((m4 + 3) * m2 ** 4 -
                              m1 ** 4 +
                              4 * m1 ** 4 -
                              6 * m1 ** 2 * orig_moments[:, 1] +
                              4 * m1 * orig_moments[:, 2])
    else:
        pass
    return orig_moments


def test_moment_matching():
    t0 = time()
    n_rv = 20
    n_scenario = 200
    data = np.random.randn(n_rv, 1000)

    tgt_moments = np.zeros((n_rv, 4))
    tgt_moments[:, 0] = data.mean(axis=1)
    tgt_moments[:, 1] = data.std(axis=1)
    tgt_moments[:, 2] = spstats.skew(data, axis=1)
    tgt_moments[:, 3] = spstats.kurtosis(data, axis=1)
    corr_mtx = np.corrcoef(data)

    out_mtx = heuristic_moment_matching(tgt_moments, corr_mtx,
                                        n_scenario=n_scenario, verbose=True)

    print "1st moments difference {}".format(
        (tgt_moments[:, 0] - out_mtx.mean(axis=1)).sum()
    )
    print "2nd moments difference {}".format(
        (tgt_moments[:, 1] - out_mtx.std(axis=1)).sum()
    )
    print "3th moments difference {}".format(
        (tgt_moments[:, 2] - spstats.skew(out_mtx, axis=1)).sum()
    )
    print "4th moments difference {}".format(
        (tgt_moments[:, 3] - spstats.kurtosis(out_mtx, axis=1)).sum()
    )

    print "corr difference {}".format(
        (corr_mtx - np.corrcoef(out_mtx)).sum()
    )
    print "HMM OK, {:.3f} secs".format(time()-t0)


if __name__ == '__main__':
    pass
    # test_moment_matching()

