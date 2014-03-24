# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''

import numpy as np
import numpy.linalg as la
import scipy.optimize as spopt
import scipy.stats as spstats


def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def cubicTransform(cubParams, EY, EX):
    '''
    cubParams: (a,b,c,d)
    EY: 4 moments of target
    EX: 12 moments of samples
    '''
#     print "cubParams;", cubParams
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

    #target origin moments, size: (n_rv * 4)
    MOM = np.zeros((n_rv, 4))
    MOM[:, 1] = 1
    MOM[:, 2] = tgtMoments[:, 2]/(tgtMoments[:, 1]**3)    #skew/(std**3)
    MOM[:, 3] = tgtMoments[:, 2]/(tgtMoments[:, 1]**4)    #skew/(std**4)
    
    #抽出moments與targetMoments相同的樣本
    #cubic transform, find good start points
    for rv in xrange(n_rv):
        cubErr, bestErr = float('inf'), float('inf')
        
        for _ in xrange(MaxStartTrial):
            tmpOut = np.random.rand(n_scenario)
            
            for _ in xrange(MaxCubIter):
                EY = MOM[rv, :]
                EX =  np.fromiter(((tmpOut**(order+1)).mean() 
                                  for order in xrange(12)), np.float)
        
                sol= spopt.root(cubicTransform, (0,1,0,0), 
                                         args=(EY, EX), method="broyden1")
                cubParams = sol.x
                root = cubicTransform(cubParams, EY, EX)
                cubErr = np.sum(np.abs(root))

                if cubErr < EPS:
                    print "early stop"
                    break
                else:
                    #update random sample(a+bx+cx^2+dx^3)
                    tmpOut = (cubParams[0] + 
                              cubParams[1]*tmpOut +
                              cubParams[2]*(tmpOut**2)+ 
                              cubParams[3]*(tmpOut**3)) 
                
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
    
    #Cholesky decomposition
    L = la.cholesky(tgtCorrMtx)  
               
    #main iteration of the algorithm
    for _ in xrange(MaxIter):
        Lp = la.cholesky(outCorrMtx)
        LpInv = la.inv(Lp)
        transMtx = L.dot(LpInv)
        tmpOutMtx = transMtx.dot(outMtx) 
        
        #update statistics
        outMoments[:, 0] = tmpOutMtx.mean(axis=1) 
        outMoments[:, 1] = tmpOutMtx.std(axis=1)
        outMoments[:, 2] = spstats.skew(tmpOutMtx, axis=1)
        outMoments[:, 3] = spstats.kurtosis(tmpOutMtx, axis=1)
        outCorrMtx = np.corrcoef(tmpOutMtx)
        
        errMoment = RMSE(outMoments, tgtMoments)
        errCorr = RMSE(outCorrMtx, tgtCorrMtx)
        
        #cubic transform
        for rv in xrange(n_rv):
            tmpOut = tmpOutMtx[rv, :]
            for _ in xrange(MaxCubIter):
                EY = MOM[rv, :]
                EX = np.fromiter(((tmpOut**(order+1)).mean() 
                                  for order in xrange(12)), np.float)
        
                sol = spopt.root(cubicTransform, np.random.rand(4), 
                                     args=(EY, EX))
        
                cubParams = sol.x 
            
                #update tmpOut y=a+bx+cx^2+dx^3
                outMtx[rv,:] = (cubParams[0] + cubParams[1]*tmpOut +
                          cubParams[2]*(tmpOut**2)+ cubParams[3]*(tmpOut**3))
    
                cubErr = RMSE(tmpOut, outMtx[rv, :])            
                if cubErr < EPS:
                    break
                else:
                    #update random sample(a+bx+cx^2+dx^3)
                    tmpOut =outMtx[rv,:]
   
        #update statistics
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
    outMtx = tgtMoments[:, 0][:, np.newaxis] + tgtMoments[:, 1][:, np.newaxis] * outMtx
    
    return outMtx 
    
if __name__ == '__main__':
    import time
    t = time.time()
    n_rv = 5
    data = np.random.randn(n_rv, 20)
    tgtMoments = np.empty((n_rv, 4))
    tgtMoments[:, 0] = data.mean(axis=1)
    tgtMoments[:, 1] = data.std(axis=1)
    tgtMoments[:, 2] = spstats.skew(data, axis=1)
    tgtMoments[:, 3] = spstats.kurtosis(data, axis=1)
    tgtCorrMtx = np.corrcoef(data)
    
    outMtx = heuristicMomentMatching(tgtMoments, tgtCorrMtx, 100)
    outMoments = np.empty((n_rv, 4))
    outMoments[:, 0] = outMtx.mean(axis=1)
    outMoments[:, 1] = outMtx.std(axis=1)
    outMoments[:, 2] = spstats.skew(outMtx, axis=1)
    outMoments[:, 3] = spstats.kurtosis(outMtx, axis=1)
    outCorrMtx = np.corrcoef(outMtx)
    
    print "error moments:", RMSE(outMoments, tgtMoments)
    print "error corrMtx:", RMSE(outCorrMtx, tgtCorrMtx)
    print "%.3f secs"%(time.time()-t)