# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import copy


def HeuristicCopula(data, alpha=0.95, n_scenario=200, K=2):
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
    empIdx = np.ceil(empData*K).astype(np.int)
    print "subcube idx:", empIdx
   
    #computing f^(d)
    delta = 1./K
    inc = 1./N/delta**(D)
    f = {}
    for arr in empIdx:
        f= SparseArray(arr, f, 0, D, inc)
    print "f1key:", f.keys()
    for key in f.keys():
        print "f2 keys:", f[key].keys()
    
    #generating samples
    for scen in xrange(n_scenario):
        sample = []
        u = np.random.rand()
        i1keys = f.keys()
        #sampling until i1 in f
        while True:
            u1 = np.random.rand()
            i1 = int(math.ceil(u1*K))
            if i1 in i1keys:
                break

        sample.append(u1)
        #sampling u2
        lowerBound = u * K**(D-1) 
        down_u2, i2 = -1, -1
        sum_f2, f2 = 0, 0
        i2keys = f[i1].keys()
        
        for kdx in xrange(1, K):
            if kdx in i2keys:
                sum_f2 += f[i1][kdx]['val']
                
            if sum_f2 >=lowerBound:
                sum_f2  -= f[i1][kdx]['val']
                down_u2 = kdx - 1
                i2 = kdx
                if i2 in i2keys:
                    f2 = f[i1][i2]['val']
        
        if down_u2 == -1 and i2 == -1:
            down_u2, i2 = K-1, K
            f2 = f[i1][i2]['val']
            
        u2 = down_u2 * delta + (u-delta**(D-1)*sum_f2)/(delta**(D-2)*f2)
        sample.append(u2)
        
        if D > 2:
            realized = [i1, i2]
            for _ in xrange(2, D+1):
                down_ud, upper_ud, sum_f = get_down_ud(realized, f, u, K)
                tmp = copy.copy(realized)
                tmp.append(upper_ud)
                ud = down_ud * delta + delta * ((u*getSparseArrayValue(realized, f)- sum_f)/( getSparseArrayValue(tmp, f)))  
                realized.append(upper_ud)
                sample.append(ud)
        print "sample:", sample
    
    
    print "HeuristicCopula_alpha-%s_scen-%s OK, %.3f secs"%(
                    alpha, n_scenario, time.time()-t0)
    

def get_down_ud(ud_arr, f, u, K):
    lowerBound = u * getSparseArrayValue(ud_arr, f)
    sum_f, down_ud, upper_ud = -1, -1, -1
    
    for k in xrange(K):
        if getSumSparseArrayValue(ud_arr, f, k) >= lowerBound:
            sum_f = getSumSparseArrayValue(ud_arr, f, k-1)
            down_ud = k-1
            upper_ud = k
    
    if down_ud == -1:
        down_ud = K-1
        upper_ud = K
        sum_f =  getSumSparseArrayValue(ud_arr, f, K-1)
        
    return down_ud, upper_ud, sum_f



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
    HeuristicCopula(data, alpha, n_scenario, K=2)



def SparseArray(arr, f, d, D, inc):
    '''
    d: current dimension
    D: maximum dimension
    '''
    if d == D:
        return
    else:
        branch = f.setdefault(arr[d], {})
        if d >= 1:
            if 'val' in branch.keys():
                branch['val'] += inc
            else: 
                branch['val'] = inc
                
        SparseArray(arr, branch, d+1, D, inc)
        return f
    
def getSparseArrayValue(arr, f, d=0):
    '''
    given array of first tree node, 
    to get the value of it in the f
    '''
    if d == len(arr):
        return f['val']
    else: 
        if arr[d] in f.keys():
            return getSparseArrayValue(arr, f[arr[d]], d+1)
        else:
            raise ValueError('%dth element of arr is %s, not in f'%(d, arr[d]))


def getSumSparseArrayValue(arr, f, upperID, d=0):
    if d == len(arr):
        return sum(f[key]['val'] for key in f.keys() if key <= upperID)
    else:
        if arr[d] in f.keys():
            return getSumSparseArrayValue(arr, f[arr[d]], upperID, d+1)
        else:
            raise ValueError('%dth element of arr is %s, not in f'%(d, arr[d]))

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
#     print getSparseArrayValue(np.array([1,3]), f)
#     print getSumSparseArrayValue(np.array([1,3]), f, 5)
# 
#     f = SparseArray(np.array([1,3,5,8, 9]), f, d, D, inc)
#     print "run2"
#     print f
