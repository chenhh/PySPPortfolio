# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

Høyland, K.; Kaut, M. & Wallace, S. W., "A heuristic for 
moment-matching scenario generation," Computational optimization 
and applications, vol. 24, pp 169-185, 2003.
'''
from __future__ import division
cimport cython
cimport numpy as np
import numpy as np
import scipy.stats as spstats
import scipy.optimize as spopt
DTYPE = np.float
ctypedef np.float_t DTYPE_t

# @cython.boundscheck(False)
cpdef HeuristicMomentMatching (np.ndarray[DTYPE_t, ndim=2]  tgtMoms, 
                               np.ndarray[DTYPE_t, ndim=2]  tgtCorrs, 
                               int n_scenario, bool verbose):
    '''
    tgtMoms, numpy.array, 1~4 central moments, size: n_rv * 4
    tgtCorrs, numpy.array, size: n_rv * n_rv
    '''
    assert tgtMoms.shape[1] == 4
    assert tgtCorrs.shape[0] ==  tgtCorrs.shape[1] == tgtMoms.shape[0]
    cdef:
        double ErrMomEPS= 1e-5
        double MaxErrMom = 1e-3
        double MaxErrCorr = 1e-3
        double cubErr, bestErr
        int n_rv = tgtMoms.shape[0]
        int MaxCubIter = 1
        int MaxMainIter = 20
        int MaxStartIter = 5
        np.ndarray[DTYPE_t, ndim=2] outMtx = np.empty((n_rv, n_scenario)) 
        np.ndarray[DTYPE_t, ndim=2] YMoms = np.zeros((n_rv, 4))
        np.ndarray[DTYPE_t, ndim=1] tmpOut = np.empty(n_scenario)
        np.ndarray[DTYPE_t, ndim=1] EY = np.empty(4)
        np.ndarray[DTYPE_t, ndim=1] EX = np.empty(12)
        
    #to generate samples Y with zero mean, and unit variance
    YMoms[:, 1] = 1
    YMoms[:, 2] = tgtMoms[:, 2]
    YMoms[:, 3] = tgtMoms[:, 3] + 3
    
    for rv in xrange(n_rv):
        cubErr = 1e40
        bestErr = 1e40

        for _ in xrange(MaxStartTrial):
            tmpOut = np.random.rand(n_scenario)
            EY = YMoms[rv, :]
       
            for _ in xrange(MaxCubIter):
                EY = MOM[rv, :]
                EX = np.array([(tmpOut**(order+1)).mean() for order in xrange(12)])

                sol= spopt.root(cubicTransform, np.zeros(4), 
                                 args=(EY, EX))
                cubParams = sol.x
                root = cubicTransform(cubParams, EY, EX)
                cubErr = np.sum(np.abs(root))
# 
                if cubErr < EPS:
                    print "early stop"
                    break
                else:
                    #update random sample(a+bx+cx^2+dx^3)
                    tmpOut = (cubParams[0] + 
                    cubParams[1]*tmpOut +
                    cubParams[2]*(tmpOut**2)+ 
                    cubParams[3]*(tmpOut**3)) 
#         
            if cubErr < bestErr:
                bestErr = cubErr
                outMtx[rv,:] = tmpOut 
    
    #computing starting properties and error
    outMoments = np.empty((n_rv, 4))
    outMoments[:, 0] = outMtx.mean(axis=1) 
    outMoments[:, 1] = outMtx.std(axis=1)
    outMoments[:, 2] = spstats.skew(outMtx, axis=1)
    outMoments[:, 3] = spstats.kurtosis(outMtx, axis=1)
    outCorrMtx = np.corrcoef(outMtx)
    
    errMoms = RMSE(outMoments, tgtMoms)
    errCorrs = RMSE(outCorrMtx, tgtCorrs)
    print 'start errMoments:%s, errCorr:%s'%(errMoms, errCorrs)



def cubicFunction(cubParams, sampleMoms, tgtMoms):
    '''
    cubParams: (a,b,c,d)
    EY: 4 moments of target
    EX: 12 moments of samples
    '''
    a, b, c, d = cubParams
    EX = sampleMoms
    EY = tgtMoms
    
    v1 = a + b*EX[0] + c*EX[1] + d*EX[2] - EY[0]
    
    v2 = ((d*d)*EX[5] + 2*c*d*EX[4] + (2*b*d + c*c)*EX[3] + 
          (2*a*d + 2*b*c)*EX[2] + (2*a*c + b*b)*EX[1] + 2*a*b*EX[0] + a*a - EY[1])
    
    v3 = ((d*d*d)*EX[8] + (3*c*d*d)*EX[7] + (3*b*d*d + 3*c*c*d)*EX[6] + 
          (3*a*d*d + 6*b*c*d + c*c*c)*EX[5] + (6*a*c*d + 3*b*b*d + 3*b*c*c)*EX[4] + 
          (a*(6*b*d + 3*c*c) + 3*b*b*c)*EX[3] + (3*a*a*d + 6*a*b*c + b*b*b) *EX[2]+ 
          (3*a*a*c + 3*a*b*b)*EX[1] + 3*a*a*b*EX[0] + a*a*a - EY[2])
    
    v4 = ((d*d*d*d)*EX[11] + (4*c*d*d*d)*EX[10] + (4*b*d*d*d + 6*c*c*d*d)*EX[9] + 
          (4*a*d*d*d + 12 *b*c*d*d + 4*c*c*c*d)*EX[8] + 
          (12*a*c*d*d + 6*b*b*d*d + 12*b*c*c*d + c*c*c*c)*EX[7] + 
          (a*(12*b*d*d + 12*c*c*d) + 12*b*b*c*d + 4*b*c*c*c)*EX[6] + 
          (6*a*a*d*d+a*(24*b*c*d+4*c*c*c)+4*b*b*b*d+6*b*b*c*c) *EX[5] + 
          (12*a*a*c*d+ a*(12*b*b*d+12*b*c*c)+4*b*b*b*c)*EX[4] + 
          (a*a*(12*b*d+6*c*c)+12*a*b*b*c+b*b*b*b)*EX[3] + 
          (4*a*a*a*d + 12*a*a*b*c+4*a*b*b*b)*EX[2] + 
          (4*a*a*a*c+6*a*a*b*b)*EX[1] + 
          (4*a*a*a*b)*EX[0] + a*a*a*a - EY[3])
    
    return v1, v2, v3, v4


cpdef errorStatistics(np.ndarray[DTYPE_t, ndim=2] outMtx, 
                    np.ndarray[DTYPE_t, ndim=2] tgtMoms, 
                    np.ndarray[DTYPE_t, ndim=2] tgtCorrs):
                    
    cdef: 
        int n_rv = outMtx.shape[0]
        np.ndarray[DTYPE_t, ndim=2] outMoms = np.empty((n_rv, 4))
        np.ndarray[DTYPE_t, ndim=2] outCorrs = np.corrcoef(outMtx)
        double errMoms = 1e50, errCorrs = 1e50 
    
    for idx in xrange(4):
        outMoms[:, idx] = (outMtx**(idx+1)).mean(axis=1)
    
    errMoms = RMSE(outMoms, tgtMoms)
    errCorrs = RMSE(outCorrs, tgtCorrs)
    return errMoms, errCorrs


cpdef RMSE(np.ndarray[DTYPE_t] srcArr, np.ndarray[DTYPE_t] tgtArr):
    '''
    srcArr, numpy.array, 1d or 2d
    tgtArr, numpy.array, 1d or 2d
    '''
    assert srcArr.shape == tgtArr.shape
    cdef double error = 1e50
    error = np.sqrt(((srcArr - tgtArr)**2).sum())
    return error 

# @cython.boundscheck(False)
cpdef cubicTransform(np.ndarray[DTYPE_t, ndim=1] cubParams, 
                   np.ndarray[DTYPE_t, ndim=1] EY, 
                   np.ndarray[DTYPE_t, ndim=1] EX):
    '''
    cubParams: (a,b,c,d)
    EY: 4 moments of target
    EX: 12 moments of samples
    '''
    cdef:
        double a, b, c, d
        double v1, v2, v3, v4
    
    a, b, c, d = cubParams
    
    v1 = a + b*EX[0] + c*EX[1] + d*EX[2] - EY[0]
    
    v2 = ((d*d)*EX[5] + 2*c*d*EX[4] + (2*b*d+c*c)*EX[3] + 
        (2*a*d+2*b*c)*EX[2] + (2*a*c+b*b)*EX[1] + 2*a*b*EX[0] + a*a - EY[1])
    
    v3 = ((d*d*d)*EX[8] + (3*c*d*d)*EX[7] + (3*b*d*d+3*c*c*d)*EX[6] + 
        (3*a*d*d+6*b*c*d+c*c*c)*EX[5] + (6*a*c*d+3*b*b*d+3*b*c*c)*EX[4] + 
        +(a*(6*b*d+3*c*c)+3*b*b*c)*EX[3] + (3*a*a*d+6*a*b*c+b*b*b) *EX[2]+ 
        (3*a*a*c+3*a*b*b)*EX[1] + 3*a*a*b*EX[0] + a*a*a - EY[2])
    
    v4 = ((d*d*d*d)*EX[11] + (4*c*d*d*d)*EX[10] + (4*b*d*d*d + 6*c*c*d*d)*EX[9] + 
        (4*a*d*d*d + 12 *b*c*d*d + 4*c*c*c*d)*EX[8] + 
        (12*a*c*d*d + 6*b*b*d*d+12*b*c*c*d+c*c*c*c)*EX[7] + 
        (a*(12*b*d*d+12*c*c*d) + 12*b*b*c*d + 4*b*c*c*c)*EX[6] + 
        (6*a*a*d*d+a*(24*b*c*d+4*c*c*c)+4*b*b*b*d+6*b*b*c*c) *EX[5] + 
        (12*a*a*c*d+ a*(12*b*b*d+12*b*c*c)+4*b*b*b*c)*EX[4] + 
        (a*a*(12*b*d+6*c*c)+12*a*b*b*c+b*b*b*b)*EX[3] + 
        (4*a*a*a*d + 12*a*a*b*c+4*a*b*b*b)*EX[2] + 
        (4*a*a*a*c+6*a*a*b*b)*EX[1] + 
        (4*a*a*a*b)*EX[0] + a*a*a*a - EY[3])
    
    return v1, v2, v3, v4




cpdef central2OrigMom(np.ndarray[DTYPE_t, ndim=2] centralMoms):
    '''
    central moments to original moments
    E[X] = samples.mean()
    std**2 = var = E[X**2] - E[X]*E[X]
    
    scipy.stats.skew, scipy.stats.kurtosis公式如下：
    m2 = np.mean((d - d.mean())**2)
    m3 = np.mean((d - d.mean())**3)
    m4 = np.mean((d - d.mean())**4)
    skew =  m3/np.sqrt(m2)**3
    kurt = m4/m2**2 -3
    '''
    cdef int n_rv = centralMoms.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] origMoms = np.empty((n_rv, 4))
    
    origMoms[:, 0] = centralMoms[:, 0]
    origMoms[:, 1] = centralMoms[:, 1] ** 2  + centralMoms[:, 0]**2 
    origMoms[:, 2] = (centralMoms[:, 2]*centralMoms[:, 1]**3+
                      centralMoms[:, 0]**3+3*centralMoms[:, 0]*centralMoms[:, 1]**2)
    origMoms[:, 3] = ((centralMoms[:, 3] + 3) * centralMoms[:, 1]**4  - centralMoms[:, 0]**4 + 
                   4*centralMoms[:, 0]**4 - 6*centralMoms[:, 0]**2*origMoms[:, 1] + 4*centralMoms[:, 0]*origMoms[:, 2])  

    return origMoms

        