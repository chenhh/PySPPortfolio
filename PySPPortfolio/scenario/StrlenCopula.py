# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

Strelen, Johann Christoph, and Feras Nassaj. "Analysis and generation of 
random vectors with copulas." Proceedings of the 39th conference on 
Winter simulation: 40 years! The best is yet to come. IEEE Press, 2007.

'''
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import copy



def StrlenHeuristicCopulaSampling(data, alpha=0.95, n_scenario=200, K=2):
    '''
    Strelen algorithm
    pure python version
    @data, numpy.array, size: N*D, N is number of data, D is the dimensions
    @alpha, float, confidence level of CVaR (0.5 <=alpha <=1)
    @n_scenario, integer
    @K, integer, num. of partitions of [0,1]
    '''
    t0 = time.time()
    N, D = data.shape

    #step 1. computing copula function
#     print "Z:", data
    
    #empirical marginal distribution of each dimension
    rank = np.empty(data.shape)
    for col in xrange(data.shape[1]):
        rank[:, col] = data[:, col].argsort() + 1
    empData = rank/data.shape[0]
#     print "U:", empData
    
    #translate U to subcube index
    empIdx = np.ceil(empData*K).astype(np.int)
#     print "subcube idx:", empIdx
   
    #computing f^(d)
    delta = 1./K
    inc = 1./N/delta**(D)
    f = {}
    for arr in empIdx:
        f= SparseArray(arr, f, 0, D, inc)
    
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
    '''
    given first node element arr, 
    get sum of the value at this level, if the key is less than or equal to upperID
    '''
    if d == len(arr):
        return sum(f[key]['val'] for key in f.keys() if key <= upperID)
    else:
        if arr[d] in f.keys():
            return getSumSparseArrayValue(arr, f[arr[d]], upperID, d+1)
        else:
            raise ValueError('%dth element of arr is %s, not in f'%(d, arr[d]))


def testStrlenHeuristicCopulaSampling():
    n_rv = 10
    data = np.random.randn(n_rv, 10)
    alpha=0.95
    n_scenario=200
    StrlenHeuristicCopulaSampling(data, alpha, n_scenario, K=1)



if __name__ == '__main__':
#     testStrlenHeuristicCopulaSampling()
    testEmpiricalCopulaCDF()
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
