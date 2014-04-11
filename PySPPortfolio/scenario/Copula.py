# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
from __future__ import division
import numpy as np
import sklearn.preprocessing as skpreprocessing
import time

def HeuristicCopula(data, alpha=0.95, n_scenario=200):
    '''
    pure python version
    @data, numpy.array, size: N*D, N is number of data, D is the dimensions
    @alpha, float, confidence level of CVaR (0.5 <=alpha <=1)
    @n_scenario, integer
    '''
    t0 = time.time()
    #step 1. computing copula function
    print "Z:", data
    
    #empirical marginal distribution of each dimension
    '''
    Z: 
      [[-0.55076287 -1.34021287  0.61119813]
       [-0.07884559  1.541239   -0.66488592]
       [-2.3594308  -0.87655953  0.39321225]
       [ 0.74696397 -1.49711055  0.62888863]
       [-0.92163626  1.41225068 -0.4675601 ]]
    U: 
      [[ 0.6  0.8  0.4]
       [ 1.   0.2  1. ]
       [ 0.2  0.6  0.6]
       [ 0.4  1.   0.2]
       [ 0.8  0.4  0.8]]
       in dimension 1, 
       the empirical dist, of r.v. less than -2.3594308 are 0.
       
    '''
    
    
    rank = np.empty(data.shape)
    for col in xrange(data.shape[1]):
        rank[:, col] = data[:, col].argsort() + 1
    empData = rank/data.shape[0]
    print "U:", empData
    
    
    print "HeuristicCopula_alpha-%s_scen-%s OK, %.3f secs"%(
                    alpha, n_scenario, time.time()-t0)
    
    
def testHeuristicCopula():
    n_rv = 5
    data = np.random.randn(n_rv, 3)
    alpha=0.95
    n_scenario=200
    HeuristicCopula(data, alpha, n_scenario)

if __name__ == '__main__':
    testHeuristicCopula()