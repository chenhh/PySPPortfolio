# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

Høyland, K.; Kaut, M. & Wallace, S. W., "A heuristic for 
moment-matching scenario generation," Computational optimization 
and applications, vol. 24, pp 169-185, 2003.
'''
from __future__ import division
import numpy as np
import numpy.linalg as la
import scipy.stats as spstats
import scipy.optimize as spopt

def HeuristicMomentMatching (tgtMoms, tgtCorrs, n_scenario=200):
    '''
    tgtMoms, numpy.array, 1~4 central moments, size: n_rv * 4
    tgtCorrs, numpy.array, size: n_rv * n_rv
    n_scenario, positive integer
    '''
    assert n_scenario >=0
    assert tgtMoms.shape[0] == tgtCorrs.shape[0] == tgtCorrs.shape[1]
    
    n_rv = tgtMoms.shape[0]
    EPS= 1e-3
    MaxErrMom = 1e-3
    MaxErrCorr = 1e-3
    
    MaxCubIter = 2
    MaxMainIter = 20
    MaxStartTrial = 20
    outMtx = np.empty((n_rv, n_scenario))
    MOM =  np.zeros((n_rv, 4)) 
    tmpOut = np.empty(n_scenario) 
    EY = np.empty(4)
    EX = np.empty(12)
    
    #origin moments, size: (n_rv * 4)
    MOM[:, 1] = 1
    MOM[:, 2] = tgtMoms[:, 2]/(tgtMoms[:, 1]**3)    #skew/(std**3)
    MOM[:, 3] = tgtMoms[:, 2]/(tgtMoms[:, 1]**4)    #skew/(std**4)
    
    for rv in xrange(n_rv):
        cubErr = float('inf')
        bestErr = float('inf')

        for _ in xrange(MaxStartTrial):
            tmpOut = np.random.rand(n_scenario)
       
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


def cubicTransform(cubParams, EY,  EX):
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


def cubicSolve(samples, probs, tgtOrigMoms):
    '''
    function that implements the Newton method
    samples, size: n_scenario, sample of a r.v.
    probs, size: n_scenario, probability of each scenario,
    tgtOrigMoms, size: 4, target original  moments
    '''
    n_trans_mom = 12
    n_orig_mom = 4
    
    #desired precision on the gradient's infinite norm
    EPSILON = 1E-12 
    START_DEV = 0.5
    MAXITER = 1000
    
    # how often we increase the START_DEV 
    IncrDevStep = 50
    currIter = 0
    sampleMoms = np.zeros(n_trans_mom)
    n_scenario = samples.size
    
    #計算samples的12階原始動差E[X]~E[X^12]
    for idx in xrange(n_trans_mom):
        sampleMoms[idx] = np.sum(probs * samples**(idx+1))
    
    #initial point of the solution
    X = np.array([0, 1, 0, 0])
    
    gradient, Hessian = GradientHessian(X, sampleMoms)
    inf_norm = la.norm(gradient)
    
    while inf_norm >= EPSILON:
        error1 = RMSE(sampleMoms, tgtOrigMoms)
        invHessian = la.inv(Hessian)
        
        #iteration of Newton method
        for idx in xrange(currIter, MAXITER):
            X = nextstep(X, gradient, invHessian)
            
            #update gradient, Hessian
            gradient, Hessian = GradientHessian(X, sampleMoms)
            inf_norm = la.norm(gradient)
            if  inf_norm < EPSILON:
                break
            invHessian = la.inv(Hessian)
            
        
        # convergence to a local maximizer or to a local minimizer greater than the initial value
        error2 = RMSE(sampleMoms, tgtOrigMoms)
        if idx < MAXITER and (error2 > error1 or isPD(invHessian) is False):
            # take a new initial point randomly generated
            X = np.array([0, 1, 0, 0]) + (np.random.rand(4) - 0.5) * START_DEV * 2**(idx/IncrDevStep)
            gradient, Hessian = GradientHessian(X, sampleMoms)
            inf_norm = la.norm(gradient)
            currIter = idx
            
    #inf_norm < EPSILON, check whether the point is a local minimizer or a local maximizer
    # If H(x) is PD, then x is a strict local minimum
    if idx < MAXITER  and isPD(Hessian) is False:
        #it was a local maximizer
        X = np.array([0, 1, 0, 0]) + (np.random.rand(4) - 0.5) * START_DEV * 2**(currIter/IncrDevStep)
        gradient, Hessian = GradientHessian(X, sampleMoms)
        inf_norm = la.norm(gradient)
        #goto while loop
    else:
        #it was a local minimizer
        error2 = RMSE(sampleMoms, tgtOrigMoms)
        
    
    return error2


def RMSE(currentMoms, tgtMoms):
    '''
    currentMoms, numpy.array, size: 4
    tgtMoms, numpy.array, size: 4
    '''
    error = np.sqrt(((currentMoms - tgtMoms)**2).sum())
    return error
    

def GradientHessian(X, currentMoms):
    '''
    computes the gradient vector and the Hessian matrix of 
    obj(X) at the current point X
    X, size: 4
    '''
    n_orig_mom=4
    gradient = np.zeros(n_orig_mom)
    Hessian = np.zeros((n_orig_mom, n_orig_mom))
        
    #Gradient at point X
    
            
   
    #hessian - no initialization needed, done for k=0
    
    
    return gradient, Hessian
   
def nextstep(X, gradient, invHessian):
    '''
    X, numpy.array, size: 4
    gradient, numpy.array, size:4
    invHessian, numpy.array, size: 4*4
    
    https://en.wikipedia.org/wiki/Newton's_method_in_optimization
    given gradient and inverse Hessian matrix, go to next step 
    x[n+1] = x[n] - f'(x)/f''(x)
    X[n+1] = X[n] - invHessian*gradient
    '''
    direction = np.multiply(invHessian, gradient)
    X -= direction
    return X
    
    
def isPD(mat):
    '''
    SPOFA tests if the matrix is positive definite
    returns True if the matrix is positive definite
    '''
    assert mat.shape[0] == mat.shape[1]
    return np.all(la.eigvals(mat) > 0)
                
  

if __name__ == '__main__':
    n_rv = 10
    data = np.random.randn(n_rv, 200)
    print data
    