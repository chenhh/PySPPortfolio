# -*- coding: utf-8 -*-
'''
Created on 2014/3/11
@author: Hung-Hsin Chen
'''
import numpy as np
import scipy.stats as spstats
import Moment
# import CMoment
import time

def testHeuristicMomentMatching():
    n_rv = 1
    n_scenario = 1000
    data = np.random.randn(n_rv, 100)
    
    moments = np.empty((n_rv, 4))
    moments[:, 0] = data .mean(axis=1)
    moments[:, 1] = data .std(axis=1)
    moments[:, 2] = spstats.skew(data , axis=1)
    moments[:, 3] = spstats.kurtosis(data , axis=1)
    corrMtx = np.corrcoef(data )
    
    if n_rv > 1:
        Moment.HeuristicMomentMatching(moments, corrMtx, n_scenario)
    else:
        Moment.HeuristicMomentMatching1D(moments, n_scenario)
#     CMoment.HeuristicMomentMatching(moments, corrMtx, n_scenario, 0)



if __name__ == '__main__':
    testHeuristicMomentMatching()
    