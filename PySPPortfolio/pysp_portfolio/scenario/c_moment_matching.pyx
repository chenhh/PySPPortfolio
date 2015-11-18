# -*- coding: utf-8 -*-
#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: infer_types=True
#cython: nonecheck=False

"""
Authors: Hung-Hsin Chen <chenhh@par.cse.nsysu.edu.tw>
License: GPL v2

Høyland, K.; Kaut, M. & Wallace, S. W., "A heuristic for
moment-matching scenario generation," Computational optimization
and applications, vol. 24, pp 169-185, 2003.

note: correlation, skewness, and kurtosis do not affected by scaling and
    shifting.
"""

from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy.optimize as spopt
import scipy.stats as spstats
from time import time

cimport numpy as cnp
from numpy.math cimport INFINITY
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.intp_t INTP_t

cpdef heuristic_moment_matching(
                              cnp.ndarray[FLOAT_t, ndim=2] tgt_moments,
                              cnp.ndarray[FLOAT_t, ndim=2] tgt_corrs,
                              int n_scenario=200,
                              int bias=True,
                              double max_moment_err=1e-3,
                              double max_corr_err=1e-3,
                              double max_cubic_err=1e-5,
                              int verbose=False):
    """
    Parameters:
    --------------
    tgt_moments:, numpy.array,shape: (n_rv * 4), 1~4 central moments
    tgt_corrs:, numpy.array, size: shape: (n_rv * n_rv), correlation matrix
    n_scenario:, positive integer, number of scenario to generate
    max_err_moment: float, max moment of error between tgt_moments and
        sample moments
    max_err_corr: float, max moment of error between tgt_corrs and
        sample correlation matrix

    Returns:
    -------------
    out_mtx: numpy.array, shape:(n_rv, n_scenario)
    """
    t0 = time()

    # parameters
    cdef:
        INTP_t n_rv = tgt_moments.shape[0]

        # iteration for find good start samples
        int max_start_iter = 5

        # cubic transform iteration
        int max_cubic_iter = 2

        # main iteration of moment matching loop
        int max_main_iter = 20

        # out mtx, for storing scenarios
        cnp.ndarray[FLOAT_t, ndim=2] out_mtx = np.zeros((n_rv, n_scenario))

        # to generate samples Y with zero mean, and unit variance,
        # shape: (n_rv, 4)
        cnp.ndarray[FLOAT_t, ndim=2] y_moments = np.zeros((n_rv, 4))

        double cubic_err, best_cub_err
        double moment_err, corrs_err
        int cub_iter, idx

        cnp.ndarray[FLOAT_t, ndim=1] ex = np.empty(4)
        cnp.ndarray[FLOAT_t, ndim=1] ey = np.empty(12)
        cnp.ndarray[FLOAT_t, ndim=1] tmp_out = np.zeros(n_scenario)
        cnp.ndarray[FLOAT_t, ndim=1] x_init
        cnp.ndarray[FLOAT_t, ndim=2] c_lower, out_corrs, co_inv, l_vec

        double ns = float(n_scenario)
        double ns_m1 = ns - 1.
        double ns_m1_2 = ns_m1 * ns_m1
        double ns_m2 = ns - 2.
        double ns_m3 = ns - 3.
        double ns2 = ns * ns

    # moments
    if bias:
        y_moments[:, 1] = 1.
        y_moments[:, 2] = tgt_moments[:, 2]
        y_moments[:, 3] = tgt_moments[:, 3] + 3.
    else:
        y_moments[:, 1] = ns_m1 / ns
        y_moments[:, 2] = (tgt_moments[:, 2] * ns_m1 * ns_m2 / ns2)
        y_moments[:, 3] = ((tgt_moments[:, 3] + 3 * ns_m1_2 / ns_m2 /
                            ns_m3) * ns_m2 * ns_m3 * ns_m1_2 / (ns2 - 1) / ns2)


    # find good start moment matrix (with err_moment converge)
    for rv in xrange(n_rv):
        cubic_err, best_cub_err = INFINITY, INFINITY

        # loop until errMom converge
        for _ in xrange(max_start_iter):
            # each random variable consists of n_scenario random sample
            tmp_out = np.random.rand(n_scenario)

            # 1~4th moments of the random variable, shape (4, )
            ey = y_moments[rv, :]

            # loop until cubic transform converge
            for cub_iter in xrange(max_cubic_iter):

                # 1~12th moments of the random samples
                ex = np.asarray([(tmp_out ** (idx + 1)).mean()
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
                    # break starter loop
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
    # correct moment, but wrong correlation
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

        # transfer mtx
        out_corrs = np.corrcoef(out_mtx)
        co_inv = la.inv(la.cholesky(out_corrs))
        l_vec = np.dot(c_lower, co_inv)
        out_mtx = np.dot(l_vec, out_mtx)

        # wrong moment, correct correlation
        moments_err, corrs_err = error_statistics(out_mtx, y_moments,
                                                  tgt_corrs)
        if verbose:
            print ('main_iter:{} cholesky transform (orig) moment_err:{}, '
                  'corr_err:{}'.format(main_iter, moments_err, corrs_err))

        # after Cholesky decompsition ,the corr_err converges,
        # but the moment error may enlarge, hence it requires
        # cubic transform
        for rv in xrange(n_rv):
            cubic_err = INFINITY
            tmp_out = out_mtx[rv, :]
            ey = y_moments[rv, :]

            # loop until cubic transform erro converge
            for cub_iter in xrange(max_cubic_iter):
                ex = np.asarray([(tmp_out ** (idx + 1)).mean()
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

    # rescale data to original moments
    out_mtx = (out_mtx * tgt_moments[:, 1][:, np.newaxis] +
               tgt_moments[:, 0][:, np.newaxis])

    cdef cnp.ndarray[FLOAT_t, ndim=2] out_central_moments = np.empty((n_rv, 4))
    out_central_moments[:, 0] = out_mtx.mean(axis=1)
    if bias:
        out_central_moments[:, 1] = out_mtx.std(axis=1)
        out_central_moments[:, 2] = spstats.skew(out_mtx, axis=1)
        out_central_moments[:, 3] = spstats.kurtosis(out_mtx, axis=1)
    else:
        out_central_moments[:, 1] = out_mtx.std(axis=1, ddof=1)
        out_central_moments[:, 2] = spstats.skew(out_mtx, axis=1, bias=False)
        out_central_moments[:, 3] = spstats.kurtosis(out_mtx, axis=1,
                                                     bias=False)


    out_corrs = np.corrcoef(out_mtx)

    if verbose:
        print ("1st moments difference {}".format(
            (tgt_moments[:, 0] - out_central_moments[:, 0]).sum())
        )
        print ("2nd moments difference {}".format(
            (tgt_moments[:, 1] - out_central_moments[:, 1]).sum())
        )
        print ("3th moments difference {}".format(
            (tgt_moments[:, 2] - out_central_moments[:, 2]).sum())
        )
        print ("4th moments difference {}".format(
            (tgt_moments[:, 3] - out_central_moments[:, 3]).sum())
        )
        print ("corr difference {}".format(
            (tgt_corrs - np.corrcoef(out_mtx)).sum())
        )

    moments_err = rmse(out_central_moments, tgt_moments)
    corrs_err = rmse(out_corrs, tgt_corrs)
    if verbose:
        print ('sample central moment err:{}, corr err:{}'.format(
            moments_err, corrs_err))

    if moments_err > max_moment_err or corrs_err > max_corr_err:
        raise ValueError("out mtx not converge, moment error: {}, "
                         "corr err:{}".format(moments_err, corrs_err))
    if verbose:
        print ("c_HeuristicMomentMatching elapsed {:.3f} secs".format(
            time() - t0))
    return out_mtx


cpdef cubic_function(cnp.ndarray[FLOAT_t, ndim=1] cubic_params,
                     cnp.ndarray[FLOAT_t, ndim=1] sample_moments,
                     cnp.ndarray[FLOAT_t, ndim=1] tgt_moments):
    """
    Parameters:
    ----------------
    cubic_params: (a,b,c,d), four floats
    sample_moments: numpy.array, shape:(12,), 1~12 moments of samples
    tgt_moments: numpy.array, shape:(4,), 1~4th moments of target
    """
    cdef:
        double a, b, c, d
        double a2, a3, a4
        double b2, b3, b4
        double c2, c3, c4
        double d2, d3, d4
        double ab,ac, ad, acd, bd, bc, bcd, cd
        cnp.ndarray[FLOAT_t, ndim=1] ex, ey
        double v1, v2, v3, v4

    a, b, c, d = cubic_params
    ex = sample_moments
    ey = tgt_moments

    a2 = a * a
    a3 = a2 * a
    a4 = a2 * a2

    b2 = b * b
    b3 = b2 * b
    b4 = b2 * b2

    c2 = c * c
    c3 = c2 * c
    c4 = c2 * c2

    d2 = d * d
    d3 = d2 * d
    d4 = d2 * d2

    ab = a * b
    ac = a * c
    ad = a * d
    acd = ac * d
    bd = b * d
    bc = b * c
    bcd = bc * d
    cd = c * d

    v1 = (a + b * ex[0] + c * ex[1] + d * ex[2] - ey[0])

    v2 = (d2 * ex[5] +
          2 * cd * ex[4] +
          (2 * bd + c2) * ex[3] +
          2 * (ad + bc) * ex[2] +
          (2 * ac + b2) * ex[1] +
          2 * ab * ex[0] +
          a2 - ey[1])

    v3 = ((d3) * ex[8] +
          (3 * c * d2) * ex[7] +
          3 * (b * d2 + c2 * d) * ex[6] +
          (3 * a * d2 + 6 * bcd + c3) * ex[5] +
          3 * (2 * acd + b2 * d + b * c2) * ex[4] +
          (a * (6 * bd + 3 * c2) + 3 * b2 * c) * ex[3] +
          (3 * a2 * d + 6 * a * bc + b3) * ex[2] +
          3 * (a2 * c + a * b2) * ex[1] +
          3 * a2 * b * ex[0] +
          a3 - ey[2])

    v4 = (d4 * ex[11] +
          (4 * cd * d2) * ex[10] +
          (4 * bd * d2 + 6 * c2 * d2) * ex[9] +
          4 * (ad * d2 + 3 * bc * d2 + c3 * d) * ex[8] +
          (12 * ac * d2 + 6 * b2 * d2 + 12 * bd * c2 + c4) * ex[7] +
          4 * (3 * ad * (bd + c2) + bc * (3 * bd + c2)) * ex[6] +
          (6 * a2 * d2 + ac * (24 * bd + 4 * c2) +
           4 * b3 * d + 6 * b2 * c2) * ex[5] +
          (12 * a2 * cd + 12 * ab * (bd + c2) + 4 * b2 * bc) * ex[4] +
          (a2 * (12 * bd + 6 * c2) + 12 * ac * b2 + b4) * ex[3] +
          4 * a * (a * ad + 3 * a * bc + b3) * ex[2] +
          a2 * (4 * ac + 6 * b2) * ex[1] +
          (4 * a2 * ab) * ex[0] +
          a4 - ey[3])

    return v1, v2, v3, v4


cdef error_statistics( cnp.ndarray[FLOAT_t, ndim=2] out_mtx,
                        cnp.ndarray[FLOAT_t, ndim=2] tgt_moments,
                        cnp.ndarray[FLOAT_t, ndim=2] tgt_corrs):
    """
    Parameters:
    ----------------
    out_mtx: numpy.array, shape: (n_rv, n_scenario)
    tgt_moments: numpy.array, shape: (n_rv, 4)
    tgt_corrs: numpy.array, shape: (n_rv, n_rv)
    """
    cdef:
        INTP_t n_rv = out_mtx.shape[0]
        cnp.ndarray[FLOAT_t, ndim=2] out_moments = np.zeros((n_rv, 4))
        cnp.ndarray[FLOAT_t, ndim=2] out_corrs = np.corrcoef(out_mtx)
        double moments_err = INFINITY, corrs_err = INFINITY
        int idx

    for idx in xrange(4):
        out_moments[:, idx] = (out_mtx ** (idx + 1)).mean(axis=1)

    moments_err = rmse(out_moments, tgt_moments)
    corrs_err = rmse(out_corrs, tgt_corrs)

    return moments_err, corrs_err


cdef rmse(cnp.ndarray[FLOAT_t, ndim=2] src_arr,
           cnp.ndarray[FLOAT_t, ndim=2] tgt_arr):
    """
    root mean square error of two arrays

    Parameters:
    ---------------------------
    src_arr: numpy.array
    tgt_arr: numpy.array
    """
    cdef double error=INFINITY
    error = np.sqrt(((src_arr - tgt_arr) ** 2).sum())
    return error

