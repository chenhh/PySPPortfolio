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


def Moment12(samples):
    n_rv = samples.shape[0]
    Moments = np.zeros((n_rv, 12))
    for order in xrange(12):
        Moments[:,order] = (samples**(order+1)).mean(axis=1)
    return Moments

def cubicTransform(param, EY, EX):
    a, b, c, d = param
    v1 = a + b*EX[0] + c*EX[1] + d*EX[2] - EY[0]
    
    v2 = (d*d)*EX[5] + 2*c*d*EX[4] + (2*b*d+c*c)*EX[3] + \
        (2*a*d+2*b*c)*EX[2] + (2*a*c+b*b)*EX[1] + 2*a*b*EX[0] + a*a - EY[1]
    
    v3 = (d*d*d)*EX[8] + (3*c*d*d)*EX[7] + (3*b*d*d+3*c*c*d)*EX[6] + \
        (3*a*d*d+6*b*c*d+c*c*c)*EX[5] + (6*a*c*d+3*b*b*d+3*b*c*c)*EX[4] + \
        +(a*(6*b*d+3*c*c)+3*b*b*c)*EX[3] + (3*a*a*d+6*a*b*c+b*b*b) *EX[2]+ \
        (3*a*a*c+3*a*b*b)*EX[1] + 3*a*a*b*EX[0] + a*a*a - EY[2]
    
    v4 = (d*d*d*d)*EX[11] + (4*c*d*d*d)*EX[10] + (4*b*d*d*d + 6*c*c*d*d)*EX[9] + \
        (4*a*d*d*d + 12 *b*c*d*d + 4*c*c*c*d)*EX[8] + \
        (12*a*c*d*d + 6*b*b*d*d+12*b*c*c*d+c*c*c*c)*EX[7] + \
        (a*(12*b*d*d+12*c*c*d) + 12*b*b*c*d + 4*b*c*c*c)*EX[6] + \
        (6*a*a*d*d+a*(24*b*c*d+4*c*c*c)+4*b*b*b*d+6*b*b*c*c) *EX[5] + \
        (12*a*a*c*d+ a*(12*b*b*d+12*b*c*c)+4*b*b*b*c)*EX[4] + \
        (a*a*(12*b*d+6*c*c)+12*a*b*b*c+b*b*b*b)*EX[3] + \
        (4*a*a*a*d + 12*a*a*b*c+4*a*b*b*b)*EX[2] + \
        (4*a*a*a*c+6*a*a*b*b)*EX[1] + \
        (4*a*a*a*b)*EX[0] + a*a*a*a - EY[3]
    return [v1, v2, v3, v4]  

def heuristicMomentMatching(targetMoments, corrMtx, n_scenario):
    '''
    given target 4 moments (mean, stdev, skewness, kurtosis)
    and correlation matrix
    @param targetMoments, numpy.array, size: n_rv * 4
    @param corrMtx, numpy.array, size: n_rv * n_rv
    @param n_scenario, positive integer
    '''
    EPSILON_Y= 1e-3
    
    assert targetMoments.shape[1] == 4
    assert targetMoments.shape[0] == corrMtx.shape[0] == corrMtx.shape[1]
    
    #generating random samples, size:(n_rv * n_scenario)
    n_rv = targetMoments.shape[0]
    X = np.random.randn(n_rv, n_scenario)
    Y = np.empty((n_rv, n_scenario))
    
    #computing 12 moments, size: (n_rv * 12)
    XMoments = Moment12(X)

    #normalized targetMoments, size: (n_rv * 4)
    MOM = np.zeros((n_rv, 4))
    MOM[:, 1] = 1   #variance =1 
    MOM[:, 2] = targetMoments[:, 2]/(targetMoments[:, 1]**3)    #skew/(std**3)
    MOM[:, 3] = targetMoments[:, 2]/(targetMoments[:, 1]**4)    #skew/(std**4)

    #cubic transform and computing Y
    for row in xrange(n_rv):
        EY = MOM[row, :]
        EX = XMoments[row, :]
        param = spopt.fsolve(cubicTransform, np.random.rand(4), args=(EY, EX), maxfev=1000)
        print "opt:", cubicTransform(param, EY, EX)
        Y[row, :] = (param[0] + param[1] * X[row, :] + param[2] * X[row, :]**2 +
                     param[3] * X[row, :]**3)
    
    #Cholesky decomposition
    L = la.cholesky(corrMtx)  
    Yp = np.dot(L, Y)
    
   
    for run in xrange(200):
        
        Rp = np.corrcoef(Yp)

        print "Rp dist:", RMSE(Rp, corrMtx)
        if(RMSE(Rp, corrMtx) <= EPSILON_Y):
            break
        Lp = la.cholesky(Rp)
        LpInv = la.inv(Lp)
        
#         print "Lp:", Lp
#         print "LpInv:", LpInv
        Yb = np.dot(LpInv, Yp)  #zero correlations, incorrect moments
        Yf = np.dot(L, Yb)      #correct correlation, incorrect moments
#         print "Yf:", Yf
        #cubic transform
        
        Moments = Moment12(Yf)
        Z = np.empty((n_rv, n_scenario))
        for row in xrange(n_rv):
            EY = MOM[row, :]
            EX = Moments[row, :]
            param = spopt.fsolve(cubicTransform, np.random.rand(4), args=(EY, EX))
            Yp[row, :] = (param[0] + param[1] * Yf[row, :] + 
                         param[2] * Yf[row, :]**2 +
                         param[3] * Yf[row, :]**3)
            print "opt:", cubicTransform(param, EY, EX)
            Z[row, :] = Yp[row, :] * targetMoments[row, 1] + targetMoments[row, 0]
        
        rMOM = np.zeros((n_rv, 4))
        rMOM[:, 0] = Z.mean(axis=1)
        rMOM[:, 1] = Z.std(axis=1)
        rMOM[:, 2] = spstats.skew(Z, axis=1)
        rMOM[:, 3] = spstats.kurtosis(Z, axis=1)
        print "moment dist:", RMSE(rMOM, targetMoments)
    
#     #restore moments
#     Z = np.empty((n_rv, n_scenario))
#     for row in xrange(n_rv):
#         Z[row, :] = Yp[row, :] * targetMoments[row, 1] + targetMoments[row, 0]
    
    
    
#     return Z 
    
if __name__ == '__main__':
    n_rv = 3
    data = np.random.randn(n_rv, 20)
    targetMoments = np.empty((n_rv, 4))
    targetMoments[:, 0] = data.mean(axis=1)
    targetMoments[:, 1] = data.std(axis=1)
    targetMoments[:, 2] = spstats.skew(data, axis=1)
    targetMoments[:, 3] = spstats.kurtosis(data, axis=1)
#     print "targetMonents:", targetMoments
    
    corrMtx = np.corrcoef(data)
    
    heuristicMomentMatching(targetMoments, corrMtx, 10)
    
    