# -*- coding: utf-8 -*-
'''
Created on 2014/3/11
@author: Hung-Hsin Chen

'''

from __future__ import division
import numpy as np
import scipy.stats as spstats
import scipy.optimize as spopt
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t






def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


@cython.boundscheck(False)
def cubicTransform(np.ndarray[DTYPE_t, ndim=1] cubParams, 
                   np.ndarray[DTYPE_t, ndim=1] EY, 
                   np.ndarray[DTYPE_t, ndim=1] EX):
    '''
    cubParams: (a,b,c,d)
    EY: 4 moments of target
    EX: 12 moments of samples
    '''
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


@cython.boundscheck(False)
def HKW_ScenGen(np.ndarray[DTYPE_t, ndim=2] tgtMoments,
                np.ndarray[DTYPE_t, ndim=2] tgtCorrMtx,
                unsigned int n_scenario):
    '''
    tgtMoments, np.array, n_rv *4
    tgtCorrMtx, np.array, n_rv * n_rv
    '''
    
    assert tgtMoments.shape[1] == 4
    assert  tgtCorrMtx.shape[0] ==  tgtCorrMtx.shape[1] == tgtMoments.shape[0]
    cdef:
        unsigned int n_rv = tgtMoments.shape[0]
        double EPS= 1e-3
        double MaxErrMoment = 1e-3
        double MaxErrCorr = 1e-3
        unsigned int MaxCubIter = 2
        unsigned int MaxIter = 20
        unsigned int MaxStartTrial = 20
        np.ndarray[DTYPE_t, ndim=2] outMtx = np.empty((n_rv, n_scenario))
        np.ndarray[DTYPE_t, ndim=2] MOM =  np.zeros((n_rv, 4)) 
        double cubErr
        double bestErr
        np.ndarray[DTYPE_t, ndim=1] tmpOut = np.empty(n_scenario) 
        np.ndarray[DTYPE_t, ndim=1] EY = np.empty(4)
        np.ndarray[DTYPE_t, ndim=1] EX = np.empty(12)
        
    #origin moments, size: (n_rv * 4)
    MOM[:, 1] = 1
    MOM[:, 2] = tgtMoments[:, 2]/(tgtMoments[:, 1]**3)    #skew/(std**3)
    MOM[:, 3] = tgtMoments[:, 2]/(tgtMoments[:, 1]**4)    #skew/(std**4)
    
    for rv in xrange(n_rv):
        cubErr = float('inf')
        bestErr = float('inf')

        for _ in xrange(MaxStartTrial):
            tmpOut = np.random.rand(n_scenario)
       
            for _ in xrange(MaxCubIter):
                EY = MOM[rv, :]
                EX = np.array([(tmpOut**(order+1)).mean() for order in xrange(12)])

                sol= spopt.root(cubicTransform, np.random.randn(4), 
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
    
    errMoment = RMSE(outMoments, tgtMoments)
    errCorr = RMSE(outCorrMtx, tgtCorrMtx)
    print 'start errMoments:%s, errCorr:%s'%(errMoment, errCorr)
