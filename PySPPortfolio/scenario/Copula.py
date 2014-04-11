# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
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
    rank = np.empty(data.shape)
    for col in xrange(data.shape[1]):
        rank[:, col] = data[:, col].argsort() + 1
    empData = rank/data.shape[0]
    print "U:", empData
    
    #將[0,1]分成K段
    K = 10
    delta = np.linspace(0, 1, K)
    
    #scatter plot
    plt.scatter(empData[:,0], empData[:,1])
    plt.show()
    
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
    n_rv = 50
    data = np.random.randn(n_rv, 2)
    alpha=0.95
    n_scenario=200
    HeuristicCopula(data, alpha, n_scenario)

if __name__ == '__main__':
#     testHeuristicCopula()
    CubicScatter()