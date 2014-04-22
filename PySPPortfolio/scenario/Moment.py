# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

Høyland, K.; Kaut, M. & Wallace, S. W., "A heuristic for 
moment-matching scenario generation," Computational optimization 
and applications, vol. 24, pp 169-185, 2003.


goal: 產生符合tgtMoms與tgtCorrs的樣本Z

Z = a*Y+b

a = sqrt(tgtMoms[1])
b = tgtMoms[0]
Y Mom[0] = 0
Y Mom[1] = 1
Y Mom[2] = tgtMoms[2]/tgtMoms[1]**3
Y Mom[3] = tgtMoms[3]/tgtMoms[1]**4



'''
from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy.optimize as spopt
import scipy.stats as spstats
import time 

def HeuristicMomentMatching (tgtMoms, tgtCorrs, n_scenario=200):
    '''
    tgtMoms, numpy.array, 1~4 central moments, size: n_rv * 4
    tgtCorrs, numpy.array, size: n_rv * n_rv
    n_scenario, positive integer
    '''
    assert n_scenario >= 0
    assert tgtMoms.shape[0] == tgtCorrs.shape[0] == tgtCorrs.shape[1]
    t0 = time.time()
    
    #parameters
    n_rv = tgtMoms.shape[0]
    ErrMomEPS= 1e-5
    MaxErrMom = 1e-3
    MaxErrCorr = 1e-3
    
    MaxCubIter = 1
    MaxMainIter = 20
    MaxStartTrial = 5
    
    #out mtx
    outMtx = np.empty((n_rv, n_scenario))
    
    #origin moments, size: (n_rv * 4)
    tgtOrigMoms = central2OrigMom(tgtMoms)
    
    #to generate samples Y with zero mean, and unit variance
    YMoms = np.zeros((n_rv, 4))
    YMoms[:, 1] = 1
    YMoms[:, 2] = tgtOrigMoms[:, 2]/tgtOrigMoms[:, 1]**3
    YMoms[:, 3] = tgtOrigMoms[:, 3]/tgtOrigMoms[:, 1]**4 
    

    #find good start matrix outMtx (with errMom converge)   
    for rv in xrange(n_rv):
        cubErr, bestCubErr = float('inf'), float('inf')

        #loop until errMom converge, but the errCorr is unreleated
        for _ in xrange(MaxStartTrial):
            #random sample
            tmpOut = np.random.randn(n_scenario)
            EY = YMoms[rv, :]
       
            #loop until ErrCubic transform converge
            for cubiter in xrange(MaxCubIter):
                EX = np.fromiter(((tmpOut**(idx+1)).mean() 
                                  for idx in xrange(12)), np.float)
                X_init = np.array([0, 1, 0, 0])
                out = spopt.leastsq(cubicFunction, X_init, args=(EX, EY), 
                                    full_output=True, ftol=1E-12, xtol=1E-12)
                cubParams = out[0] 
                cubErr = np.sum(out[2]['fvec']**2)
       
                tmpOut = (cubParams[0] +  cubParams[1]*tmpOut +
                          cubParams[2]*(tmpOut**2) + cubParams[3]*(tmpOut**3))
               
                if cubErr < ErrMomEPS:
                    break
                else:
                    print "rv:%s, cubiter:%s, cubErr: %s, not converge"%(rv, cubiter, cubErr)
         
            #accept current samples
            if cubErr < bestCubErr:
                bestCubErr = cubErr
                outMtx[rv,:] = tmpOut 
            
    #computing starting properties and error
    errMoms, errCorrs = errorStatistics(outMtx, YMoms, tgtCorrs)
    print 'start mtx errMom:%s, errCorr:%s'%(errMoms, errCorrs)

    #Cholesky decomp of target corr mtx
    outCorrs = np.corrcoef(outMtx)
    C = la.cholesky(tgtCorrs)
    
    #main iteration of HKW
    for mainIter in xrange(MaxMainIter):
        if errMoms < MaxErrMom and errCorrs < MaxErrCorr:
            break
        
        #transfer mtx
        CO_inv = la.inv(la.cholesky(outCorrs))
        L = np.dot(C, CO_inv)
        outMtx = np.dot(L, outMtx)
        
        errMoms, errCorrs = errorStatistics(outMtx, tgtMoms, tgtCorrs)
        print 'mainIter:%s errMom:%s, errCorr:%s'%(mainIter, errMoms, errCorrs)
    
        #cubic transform
        for rv in xrange(n_rv):
            cubErr, bestCubErr = float('inf'), float('inf')
            
            tmpOut = outMtx[rv, :]
            EY = tgtOrigMoms[rv, :]
            
            #loop until ErrCubic transform converge
            for cubiter in xrange(MaxCubIter):
                EX = np.fromiter(((tmpOut**(idx+1)).mean() 
                                  for idx in xrange(12)), np.float)
                X_init = np.array([0, 1, 0, 0])
                out = spopt.leastsq(cubicFunction, X_init, args=(EX, EY), 
                                    full_output=True, ftol=1E-12, xtol=1E-12)
                cubParams = out[0] 
                cubErr = np.sum(out[2]['fvec']**2)
       
                tmpOut = (cubParams[0] +  cubParams[1]*tmpOut +
                          cubParams[2]*(tmpOut**2) + cubParams[3]*(tmpOut**3))
               
                if cubErr < ErrMomEPS:
                    outMtx[rv, :] = tmpOut
                    break
                else:
                    print "mainIter, rv:%s, cubiter:%s, cubErr: %s, not converge"%(rv, cubiter, cubErr)
                
        errMoms, errCorrs = errorStatistics(outMtx, tgtMoms, tgtCorrs)
        print 'mainIter cubicTransform: %s errMom:%s, errCorr:%s'%(mainIter, errMoms, errCorrs)
        
        print "elapsed %.3f secs"%(time.time()-t0)

def cubicSolve(samples, probs, tgtMoms):
    '''
    http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.leastsq.html
    using scipy sum of square function to get cubicParam
    
    '''
    assert tgtMoms.size == 4
    n_origMom = 12  
    sampleMoms = np.zeros(n_origMom)
    
    #計算samples的12階原始動差E[X]~E[X^12]
    sampleMoms = np.fromiter( (np.sum(probs* samples**(idx+1) 
                            for idx in xrange(n_origMom))), dtype=np.float)
    # initial guess 
    X_init = np.array([0, 1, 0, 0])
    
    #If ier is equal to 1, 2, 3 or 4, the solution was found.
    out = spopt.leastsq(cubicFunction, X_init, args=(sampleMoms, tgtMoms), full_output=True, ftol=1E-12)
    cubParam = out[0]
    msg = out[2]
    lsq = sum(msg['fvec'])
    return cubParam, lsq
    

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

def errorStatistics(outMtx, tgtMoms, tgtCorrs):
    n_rv = outMtx.shape[0]
    outMoms = np.empty((n_rv, 4))
    outMoms[:, 0] = outMtx.mean(axis=1) 
#     outMoms[:, 1] = outMtx.std(axis=1)
#     outMoms[:, 2] = spstats.skew(outMtx, axis=1)
#     outMoms[:, 3] = spstats.kurtosis(outMtx, axis=1)
    outMoms[:, 1] = (outMtx**2).mean(axis=1)
    
  
    outCorrMtx = np.corrcoef(outMtx)
    
    errMoms = RMSE(outMoms, tgtMoms)
    errCorrs = RMSE(outCorrMtx, tgtCorrs)
    return errMoms, errCorrs


def RMSE(srcArr, tgtArr):
    '''
    srcArr, numpy.array
    tgtArr, numpy.array
    '''
    assert srcArr.shape == tgtArr.shape
    error = np.sqrt(((srcArr - tgtArr)**2).sum())
    return error  

def central2OrigMom(centralMoms):
    '''
    E[X] = samples.mean()
    var = E[X**2] - E[X]*E[X]
    skew =  np.mean((samples - samples.mean())**3)/Moms[1]**3
    kurt =  np.mean((samples - samples.mean())**4)/Moms[1]**4 -3 
         = (E[X**4] - 4*u*E[X**3] + 6*u**2*E[X**4] - 4*u**3*E[X]+u**4)/std**4-3
    '''
    n_rv = centralMoms.shape[0]
    origMoms = np.empty((n_rv, 4))
    origMoms[:, 0] = centralMoms[:, 0]
    origMoms[:, 1] = centralMoms[:, 1] ** 2  + centralMoms[:, 0]**2 
    origMoms[:, 2] = (centralMoms[:, 2]*centralMoms[:, 1]**3+
                      centralMoms[:, 0]**3+3*centralMoms[:, 0]*centralMoms[:, 1]**2)
    origMoms[:, 3] = ((centralMoms[:, 3] + 3) * centralMoms[:, 1]**4  - centralMoms[:, 0]**4 + 
                   4*centralMoms[:, 0]**4 - 6*centralMoms[:, 0]**2*origMoms[:, 1] + 4*centralMoms[:, 0]*origMoms[:, 2])  
#     print "origMom:", origMoms
    return origMoms
    

# def cubicSolve(samples, probs, tgtMoms):
#     '''
#     function that implements the Newton method
#     @samples, size: n_scenario, sample of a r.v.
#     @probs, size: n_scenario, probability of each scenario,
#     @tgtMoms, size: 4, target original  moments
#     
#     @return cubicParam, numpy.array, size:4 (a,b,c,d)
#     
#     '''
#     n_origMom = 12
#     n_tgtMom = 4
#     
#     #desired precision on the gradient's infinite norm
#     EPSILON = 1E-12 
#     START_DEV = 0.5
#     MAXITER = 1000
#     
#     # how often we increase the START_DEV 
#     IncrDevStep = 50
#     currIter = 0
#     sampleMoms = np.zeros(n_origMom)
#     n_scenario = samples.size
#     
#     #計算samples的12階原始動差E[X]~E[X^12]
#     for idx in xrange(n_origMom):
#         sampleMoms[idx] = np.sum(probs * samples**(idx+1))
#     
#     #initial point of the solution
#     X = np.array([0, 1, 0, 0])
#     
#     gradient, Hessian = GradientHessian(X, sampleMoms)
#     inf_norm = la.norm(gradient, np.inf)
#     
#     while inf_norm >= EPSILON:
#         error1 = RMSE(sampleMoms, tgtMoms)
#         invHessian = la.inv(Hessian)
#         
#         #iteration of Newton method
#         for idx in xrange(currIter, MAXITER):
#             X = nextstep(X, gradient, invHessian)
#             
#             #update gradient, Hessian
#             gradient, Hessian = GradientHessian(X, sampleMoms)
#             inf_norm = la.norm(gradient, np.inf)
#             if  inf_norm < EPSILON:
#                 break
#             invHessian = la.inv(Hessian)
#             
#         
#         # convergence to a local maximizer or to a local minimizer greater than the initial value
#         error2 = RMSE(sampleMoms, tgtMoms)
#         if idx < MAXITER and (error2 > error1 or isPD(invHessian) is False):
#             # take a new initial point randomly generated
#             X = np.array([0, 1, 0, 0]) + (np.random.rand(4) - 0.5) * START_DEV * 2**(idx/IncrDevStep)
#             gradient, Hessian = GradientHessian(X, sampleMoms)
#             inf_norm = la.norm(gradient, np.inf)
#             currIter = idx
#             
#     #inf_norm < EPSILON, check whether the point is a local minimizer or a local maximizer
#     # If H(x) is PD, then x is a strict local minimum
#     if idx < MAXITER  and isPD(Hessian) is False:
#         #it was a local maximizer
#         X = np.asarray([0, 1, 0, 0]) + (np.random.rand(4) - 0.5) * START_DEV * 2**(currIter/IncrDevStep)
#         gradient, Hessian = GradientHessian(X, sampleMoms)
#         inf_norm = la.norm(gradient, np.inf)
#         #goto while loop
#     else:
#         #it was a local minimizer
#         error2 = RMSE(sampleMoms, tgtMoms)
#         
#     
#     return error2
# 
# 
# def RMSE(currentMoms, tgtMoms):
#     '''
#     currentMoms, numpy.array, size: 4
#     tgtMoms, numpy.array, size: 4
#     '''
#     error = np.sqrt(((currentMoms - tgtMoms)**2).sum())
#     return error
#     
# 
# def GradientHessian(X, currentMoms):
#     '''
#     Minimum of the sum of squares
#     computes the gradient vector and the Hessian matrix of 
#     obj(X) at the current point X
#     X, size: 4
#     '''
#     n_orig_mom=4
#     gradient = np.zeros(n_orig_mom)
#     Hessian = np.zeros((n_orig_mom, n_orig_mom))
#         
#     #Gradient at point X
#     
#             
#    
#     #hessian - no initialization needed, done for k=0
#     
#     
#     return gradient, Hessian
#    
# def nextstep(X, gradient, invHessian):
#     '''
#     X, numpy.array, size: 4
#     gradient, numpy.array, size:4
#     invHessian, numpy.array, size: 4*4
#     
#     https://en.wikipedia.org/wiki/Newton's_method_in_optimization
#     given gradient and inverse Hessian matrix, go to next step 
#     x[n+1] = x[n] - f'(x)/f''(x)
#     X[n+1] = X[n] - invHessian*gradient
#     '''
#     direction = np.multiply(invHessian, gradient)
#     X -= direction
#     return X
#     
#     
# def isPD(mat):
#     '''
#     SPOFA tests if the matrix is positive definite
#     returns True if the matrix is positive definite
#     '''
#     assert mat.shape[0] == mat.shape[1]
#     return np.all(la.eigvals(mat) > 0)
                

def testHMM():
    n_rv = 3
    n_scenario = 200
    data = np.random.randn(n_rv, 100)

    Moms =  np.zeros((n_rv, 4))    
    Moms[:, 0] = data.mean(axis=1)
    Moms[:, 1] = data.std(axis=1)
    Moms[:, 2] = spstats.skew(data, axis=1)
    Moms[:, 3] = spstats.kurtosis(data, axis=1)
    corrMtx = np.corrcoef(data)
    HeuristicMomentMatching(Moms, corrMtx, n_scenario=200)


# def testCentral2OrigMom():
#     n_scenario = 100
#     samples = np.random.randn(n_scenario)
#     origMoms = np.fromiter( ((samples**(idx+1)).mean() 
#                             for idx in xrange(4)), dtype=np.float)
#    
#    
#     #skew*Moms[1]**3+Moms[0]**3+3*Moms[0]*Moms[1]**2
#     
# 
#     central2OrigMom(Moms, n_scenario)


def testScale():
    a = np.random.randn(100)
    print "a mean:", a.mean()
    print "a std:", a.std()
    print "a skew:", spstats.skew(a)
    print "a kurt:", spstats.kurtosis(a)
    
    b = (a - a.mean())/a.std()
    print "b mean:", b.mean()
    print "b std:", b.std()
    print "b skew:", spstats.skew(b)
    print "b kurt:", spstats.kurtosis(b)
    
    c = b*8
    print "c mean:", c.mean()
    print "c std:", c.std()
    print "c skew:", spstats.skew(c)
    print "c kurt:", spstats.kurtosis(c)
    
    
    d = b*8 + 10
    print "d mean:", d.mean()
    print "d std:", d.std()
    print "d skew:", spstats.skew(d)
    print "d kurt:", spstats.kurtosis(d)
    
if __name__ == '__main__':
#     testCentral2OrigMom()
#    testHMM()
    testScale()
   
