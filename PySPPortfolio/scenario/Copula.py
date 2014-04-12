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
    for arr in empIdx:
        f= SparseArray(arr, f, 0, D, inc)
    print "fkey:", f.keys()
    
    #generating samples
    for scen in xrange(n_scenario):
        u = np.random.rand()
        u1 = np.random.rand()
        down_u2_idx = down_u2(u, u1, K, D, f)
        u2 = down_u2_idx * delta 
        
        
        if D > 2:
            pass
    
    
    
    print "HeuristicCopula_alpha-%s_scen-%s OK, %.3f secs"%(
                    alpha, n_scenario, time.time()-t0)
    


def down_u2(u, u1, K, D, f):
    lowBound = K**(D-1) * u
    i1 = int(np.maximum(1, np.ceil(u1*K)))
    fsum, best_u2 = 0, 0
    for val2 in xrange(K):
        i2 = int(val2)
        if i2 in f[i1].keys():
            fsum += f[i1][i2]['val']
        
        if fsum >= lowBound:
            best_u2 = i2 -1
            
    return best_u2
        
    
    
    return best

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



def SparseArray(arr, f, d, D, inc):
    '''
    d: current dimension
    D: maximum dimension
    '''
    if d == D:
        return
    else:
        branch = f.setdefault(int(arr[d]), {})
        if d >= 1:
            if 'val' in branch.keys():
                branch['val'] += inc
            else: 
                branch['val'] = inc
                
        SparseArray(arr, branch, d+1, D, inc)
        return f
    


if __name__ == '__main__':
    testHeuristicCopula()
#     CubicScatter()
    
#     arr = np.array([1,3,5,7,9])
#     d = 0
#     D = 4
#     inc = 10
#     f = SparseArray(arr, {}, d, D, inc)
#     print "run1"
#     print f
# 
#     f = SparseArray(np.array([1,3,5,8, 9]), f, d, D, inc)
#     print "run2"
#     print f
