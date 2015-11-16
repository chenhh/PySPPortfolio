# -*- coding: utf-8 -*-
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

cpdef heuristic_moment_matching(
                              cnp.ndarray[FLOAT_t, ndim=2] tgt_moments,
                              cnp.ndarray[FLOAT_t, ndim=2] tgt_corrs,
                              int n_scenario=200,
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
        int n_rv = tgt_moments.shape[0]

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
        int cub_iter

        cnp.ndarray[FLOAT_t, ndim=1] ex = np.empty(4)
        cnp.ndarray[FLOAT_t, ndim=1] ey = np.empty(12)
        cnp.ndarray[FLOAT_t, ndim=1] tmp_out = np.zeros(n_scenario)
        cnp.ndarray[FLOAT_t, ndim=1] x_init
        cnp.ndarray[FLOAT_t, ndim=2] c_lower, out_corrs, co_inv, l_vec

    # moments
    y_moments[:, 1] = 1
    y_moments[:, 2] = tgt_moments[:, 2]
    y_moments[:, 3] = tgt_moments[:, 3] + 3


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
    out_central_moments[:, 1] = out_mtx.std(axis=1)
    out_central_moments[:, 2] = spstats.skew(out_mtx, axis=1)
    out_central_moments[:, 3] = spstats.kurtosis(out_mtx, axis=1)
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
        cnp.ndarray[FLOAT_t, ndim=1] ex, ey
        double v1, v2, v3, v4

    a, b, c, d = cubic_params
    ex = sample_moments
    ey = tgt_moments

    v1 = (a + b * ex[0] + c * ex[1] + d * ex[2] - ey[0])

    v2 = ((d * d) * ex[5] +
          2 * c * d * ex[4] +
          (2 * b * d + c * c) * ex[3] +
          (2 * a * d + 2 * b * c) * ex[2] +
          (2 * a * c + b * b) * ex[1] +
          2 * a * b * ex[0] +
          a * a - ey[1])

    v3 = ((d * d * d) * ex[8] +
          (3 * c * d * d) * ex[7] +
          (3 * b * d * d + 3 * c * c * d) * ex[6] +
          (3 * a * d * d + 6 * b * c * d + c * c * c) * ex[5] +
          (6 * a * c * d + 3 * b * b * d + 3 * b * c * c) * ex[4] +
          (a * (6 * b * d + 3 * c * c) + 3 * b * b * c) * ex[3] +
          (3 * a * a * d + 6 * a * b * c + b * b * b) * ex[2] +
          (3 * a * a * c + 3 * a * b * b) * ex[1] +
          3 * a * a * b * ex[0] +
          a * a * a - ey[2])

    v4 = ((d * d * d * d) * ex[11] +
          (4 * c * d * d * d) * ex[10] +
          (4 * b * d * d * d + 6 * c * c * d * d) * ex[9] +
          (4 * a * d * d * d + 12 * b * c * d * d + 4 * c * c * c * d) * ex[8] +
          (
              12 * a * c * d * d + 6 * b * b * d * d + 12 * b * c * c * d + c * c * c * c) *
          ex[7] +
          (a * (
              12 * b * d * d + 12 * c * c * d) + 12 * b * b * c * d + 4 * b * c * c * c) *
          ex[6] +
          (6 * a * a * d * d + a * (
              24 * b * c * d + 4 * c * c * c) + 4 * b * b * b * d + 6 * b * b * c * c) *
          ex[5] +
          (12 * a * a * c * d + a * (
              12 * b * b * d + 12 * b * c * c) + 4 * b * b * b * c) * ex[4] +
          (a * a * (
              12 * b * d + 6 * c * c) + 12 * a * b * b * c + b * b * b * b) *
          ex[
              3] +
          (4 * a * a * a * d + 12 * a * a * b * c + 4 * a * b * b * b) * ex[2] +
          (4 * a * a * a * c + 6 * a * a * b * b) * ex[1] +
          (4 * a * a * a * b) * ex[0] +
          a * a * a * a - ey[3])

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
        int n_rv = out_mtx.shape[0]
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


cdef central_to_orig_moment(cnp.ndarray[FLOAT_t, ndim=2] central_moments):
    """
    central moments to original moments

    Parameters:
    ------------------
    central_monents: numpy.array, shape:(n_data, 4)

    -----------------------
    E[X] = samples.mean()
    std**2 = var = E[X**2] - E[X]*E[X]

    scipy.stats.skew, scipy.stats.kurtosis equations:
    m2 = np.mean((d - d.mean())**2)
    m3 = np.mean((d - d.mean())**3)
    m4 = np.mean((d - d.mean())**4)
    skew =  m3/np.sqrt(m2)**3
    kurt = m4/m2**2 -3
    """
    cdef:
        int n_rv = central_moments.shape[0]
        cnp.ndarray[FLOAT_t, ndim=2] orig_moments = np.empty((n_rv, 4))

    orig_moments[:, 0] = central_moments[:, 0]

    orig_moments[:, 1] = (central_moments[:, 1] ** 2
                          + central_moments[:, 0] ** 2)

    orig_moments[:, 2] = (central_moments[:, 2] *
                          central_moments[:, 1] ** 3 +
                          central_moments[:, 0] ** 3 +
                          3 * central_moments[:, 0] *
                          central_moments[:, 1] ** 2)
    orig_moments[:, 3] = ((central_moments[:, 3] + 3) *
                          central_moments[:, 1] ** 4 -
                          central_moments[:, 0] ** 4 +
                          4 * central_moments[:, 0] ** 4 -
                          6 * central_moments[:, 0] ** 2 *
                          orig_moments[:, 1] +
                          4 * central_moments[:, 0] *
                          orig_moments[:, 2])

    return orig_moments
#
