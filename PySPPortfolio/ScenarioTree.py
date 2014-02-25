# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''

import numpy as np
import scipy as sp
import scipy.stats as spstats

def heuristicMomentMatching(targetMoments, corrMtx, n_scenario):
    '''
    given target 4 moments (mean, stdev, skewness, kurtosis)
    and correlation matrix
    @param targetMoments, numpy.array, size: n_rv * 4
    @param corrMtx, numpy.array, size: n_rv * n_rv
    @param n_scenario, positive integer
    
    '''
    assert targetMoments.shape[1] == 4
    assert targetMoments.shape[0] == corrMtx.shape[0] == corrMtx.shape[1]
    
    #generating random samples
    n_rv = targetMoments.shape[0]
    X = np.random.randn((n_rv, n_scenario))

    #computing 12 moments
    XMoments = np.zeros((n_rv, 12))
    for order in xrange(12):
        XMoments[:,order] = (X**(order+1)).mean(axis=1)

    #normalized targetMoments
    MOM = np.zeros((n_rv, 4))
    MOM[:, 1] = 1   #variance =1 
    MOM[:, 2] = targetMoments[:, 2]/(targetMoments[:, 1]**3)    #skew/(std**3)
    MOM[:, 3] = targetMoments[:, 2]/(targetMoments[:, 1]**4)    #skew/(std**4)

    #cubic transform
    
    
    
if __name__ == '__main__':
    n_rv = 10
    data = np.random.randn(n_rv, 20)
    targetMoments = np.empty((n_rv, 4))
    targetMoments[:, 0] = data.mean(axis=1)
    targetMoments[:, 1] = data.std(axis=1)
    targetMoments[:, 2] = spstats.skew(data, axis=1)
    targetMoments[:, 3] = spstats.kurtosis(data, axis=1)
#     print "targetMonents:", targetMoments
    
    corrMtx = np.corrcoef(data)
    
    heuristicMomentMatching(targetMoments, corrMtx, n_scenario=10)
    
    