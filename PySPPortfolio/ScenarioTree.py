# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''

import numpy as np
import numpy.linalg as la
import scipy.optimize as spopt
import scipy.stats as spstats
from openopt import NLP

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def Moment12(samples):
    n_rv = samples.shape[0]
    Moments = np.zeros((n_rv, 12))
    for order in xrange(12):
       4 Moments[:,order] = (samples**(order+1)).mean(axis=1)
    return Moments



def cubicTransform(param, EY, EX):
    print "param;", param
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
    
    return v1+v2+v3+v4

def heuristicMomentMatching(tgtMoments, tgtCorrMtx, n_scenario):
    '''
    given target 4 moments (mean, stdev, skewness, kurtosis)
    and correlation matrix
    @param tgtMoments, numpy.array, size: n_rv * 4
    @param tgtCorrMtx, numpy.array, size: n_rv * n_rv
    @param n_scenario, positive integer
    '''
    EPS= 1e-3
    MaxErrMoment = 1e-3
    MaxErrCorr = 1e-3
    MaxCubIter = 2
    MaxIter = 20
    MaxStartTrial = 20
    
    assert tgtMoments.shape[1] == 4
    assert tgtMoments.shape[0] == tgtCorrMtx.shape[0] == tgtCorrMtx.shape[1]
    
    n_rv = tgtMoments.shape[0]
    outMtx = np.empty((n_rv, n_scenario))

    #original target moments, size: (n_rv * 4)
    MOM = np.zeros((n_rv, 4))
    MOM[:, 1] = 1
    MOM[:, 2] = tgtMoments[:, 2]/(tgtMoments[:, 1]**3)    #skew/(std**3)
    MOM[:, 3] = tgtMoments[:, 2]/(tgtMoments[:, 1]**4)    #skew/(std**4)

    #抽出moments與targetMoments相同的樣本(但corr. mtx不同)
    #cubic transform
    for rv in xrange(n_rv):
        cubErr, bestErr = float('inf'), float('inf')
        
        for start_trial in xrange(MaxStartTrial):
            #random samples
            TmpOut = np.random.randn(n_rv)
            
            for cubIter in xrange(MaxCubIter):
                EY = MOM[rv, :]
                EX = moment12ofTmpOut
        
                param = spopt.fsolve(cubicTransform, np.random.rand(4), args=(EY, EX), maxfev=10000)
        
                cubErr = (param, EX, EY)
                
                if cubErr < EPS:
                    break
                else:
                    #update random sample(a+bx+cx^2+dx^3)
                    TmpOut = (param, EX)
                
            if cubErr < bestErr:
                bestErr = cubErr
                outMtx[n_rv,:] = (param, TmpOut)
    
    #computing starting properties and error
    outMoments = np.empty((n_rv, 4))
    outMoments[:, 0] = outMtx.mean(axis=1) 
    outMoments[:, 1] = outMtx.std(axis=1)
    outMoments[:, 2] = spstats.skew(outMtx, axis=1)
    outMoments[:, 3] = spstats.kurtosis(outMtx, axis=1)
    outCorrMtx = np.corrcoef(outMtx)
    
    errMoment = RMSE(outMoments, tgtMoments)
    errCorr = RMSE(outCorrMtx, tgtCorrMtx)
    
    #Cholesky decomposition
    L = la.cholesky(tgtCorrMtx)  
               
    #main iteration of the algorithm
    for iter in xrange(MaxIter):
        LOut = la.cholesky(outCorrMtx)
        LOutInv = la.inv(LOut)
        transMtx = L.dot(LOutInv)
        TmpOutMtx = transMtx.dot(outMtx) 
        
        #update statistics
        outMoments[:, 0] = TmpOutMtx.mean(axis=1) 
        outMoments[:, 1] = TmpOutMtx.std(axis=1)
        outMoments[:, 2] = spstats.skew(TmpOutMtx, axis=1)
        outMoments[:, 3] = spstats.kurtosis(TmpOutMtx, axis=1)
        outCorrMtx = np.corrcoef(TmpOutMtx)
        
        errMoment = RMSE(outMoments, tgtMoments)
        errCorr = RMSE(outCorrMtx, tgtCorrMtx)
        
        #cubic transform
        for rv in xrange(n_rv):
            for cubIter in xrange(MaxCubIter):
                EY = MOM[rv, :]
                EX = moment12ofTmpOutMtx
        
                param = spopt.fsolve(cubicTransform, np.random.rand(4), args=(EY, EX), maxfev=10000)
        
        
                cubErr = (param, EX, EY)
                
                if cubErr < EPS:
                    break
                else:
                    #update random sample(a+bx+cx^2+dx^3)
                    OutMtx = (param, EX)
   
        outMoments[:, 0] = outMtx.mean(axis=1) 
        outMoments[:, 1] = outMtx.std(axis=1)
        outMoments[:, 2] = spstats.skew(outMtx, axis=1)
        outMoments[:, 3] = spstats.kurtosis(outMtx, axis=1)
        outCorrMtx = np.corrcoef(outMtx)
    
        errMoment = RMSE(outMoments, tgtMoments)
        errCorr = RMSE(outCorrMtx, tgtCorrMtx)
        
        
        if errMoment <= MaxErrMoment and errCorr <= MaxErrCorr:
            break
            
    #rescale samples    
    outMtx = tgtMoments[:, 0] + tgtMoments[:, 1] * outMtx
    
    return outMtx 
    
if __name__ == '__main__':
    n_rv = 5
    data = np.random.randn(n_rv, 20)
    targetMoments = np.empty((n_rv, 4))
    targetMoments[:, 0] = data.mean(axis=1)
    targetMoments[:, 1] = data.std(axis=1)
    targetMoments[:, 2] = spstats.skew(data, axis=1)
    targetMoments[:, 3] = spstats.kurtosis(data, axis=1)
    print "targetMonents:"
    print targetMoments.T
    
    print "corr"
    corrMtx = np.corrcoef(data)
    print corrMtx
#     heuristicMomentMatching(targetMoments, corrMtx, 10)
    
    