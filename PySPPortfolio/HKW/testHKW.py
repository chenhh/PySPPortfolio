# -*- coding: utf-8 -*-
'''
Created on 2014/3/11
@author: Hung-Hsin Chen

'''

import numpy as np
import scipy.stats as spstats
from time import time
import HKW

n_rv = 5
scen= 20
data = np.random.randn(n_rv, scen)
tgtMoments = np.empty((n_rv, 4))
tgtMoments[:, 0] = data.mean(axis=1)
tgtMoments[:, 1] = data.std(axis=1)
tgtMoments[:, 2] = spstats.skew(data, axis=1)
tgtMoments[:, 3] = spstats.kurtosis(data, axis=1)
tgtCorrMtx = np.corrcoef(data)

t = time()
HKW.HKW_ScenGen(tgtMoments, tgtCorrMtx, scen)
print "%.3f secs"%(time()-t)
if __name__ == '__main__':
    pass