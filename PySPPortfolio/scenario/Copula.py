# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time


def HeuristicCopula(data, alpha=0.95, n_scenario=200, K=100):
    '''
    pure python version
    @data, numpy.array, size: N*D, N is number of data, D is the dimensions
    @alpha, float, confidence level of CVaR (0.5 <=alpha <=1)
    @n_scenario, integer
    @K, integer, num. of partitions of [0,1]
    '''
    t0 = time.time()
    N, D = data.shape

    #step 1. computing copula function
    print "Z:", data
    
    #empirical marginal distribution of each dimension
    rank = np.empty(data.shape)
    for col in xrange(data.shape[1]):
        rank[:, col] = data[:, col].argsort() + 1
    empData = rank/data.shape[0]
    print "U:", empData
    
    #translate U to subcube index
    empIdx = np.maximum(np.ceil(empData*K), 1)
    print "subcube idx:", empIdx
   
    #computing f^(d)
    delta = 1./K
    inc = 1./N/delta**(D)
    f = {}
    
    
    
    print "HeuristicCopula_alpha-%s_scen-%s OK, %.3f secs"%(
                    alpha, n_scenario, time.time()-t0)
    
   
def CubicScatter():
    from numpy.random import random
    from mpl_toolkits.mplot3d import Axes3D
    
    colors=['b', 'c', 'y', 'm', 'r']
    
    ax = plt.subplot(111, projection='3d')
    
    ax.plot(random(10), random(10), random(10), 'x', color=colors[0], label='Low Outlier')
    ax.plot(random(10), random(10), random(10), 'o', color=colors[0], label='LoLo')
    ax.plot(random(10), random(10), random(10), 'o', color=colors[1], label='Lo')
    ax.plot(random(10), random(10), random(10), 'o', color=colors[2], label='Average')
    ax.plot(random(10), random(10), random(10), 'o', color=colors[3], label='Hi')
    ax.plot(random(10), random(10), random(10), 'o', color=colors[4], label='HiHi')
    ax.plot(random(10), random(10), random(10), 'x', color=colors[4], label='High Outlier')
    
    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
    
    plt.show()
    
def testHeuristicCopula():
    n_rv = 10
    data = np.random.randn(n_rv, 40)
    alpha=0.95
    n_scenario=200
    HeuristicCopula(data, alpha, n_scenario, K=100)


def SparseTree(arr, f, d, D):
    '''
    d: current dimension
    D: maximum dimension
    '''
    if d == D:
        return
    else:
        print "d:",d
        branch = f.setdefault(arr[d], {})
        if d >= 1:
            branch['val'] = 0.01 
            
        print "f:", f, id(f)
        print "branch:", branch, id(branch)
        SparseTree(arr, branch, d+1, D)
#         print "f:", f
        return f
    


if __name__ == '__main__':
#     testHeuristicCopula()
#     CubicScatter()
    
    arr = np.array([1,3,5,7])
    d = 0
    D = 4
    f = SparseTree(arr, {}, d, D)
    print
    print f

    f = SparseTree(np.array([1,2,5,7]), f, d, D)
    print 
    print f