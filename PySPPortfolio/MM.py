# -*- coding: utf-8 -*-
'''
Created on 2014/4/29

@author: Hung-Hsin Chen
'''

from __future__ import division
from datetime import date
import os
import pandas as pd
import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt
import scenario.Moment as mom

PklBasicFeaturesDir = os.path.join(os.getcwd(),'pkl', 'BasicFeatures')
symbols = ('2330', '1216')
startDate, endDate = date(2005, 1, 1), date(2013, 12, 31)

def readDF(symbols):
    dfs = []
    for symbol in symbols:
        fin = os.path.join(PklBasicFeaturesDir, "%s.pkl"%(symbol))
        dfs.append(pd.read_pickle(fin))
    
    return dfs

def plotDistribution(dfs, today, histPeriods = 20, n_scenario = 200):
    '''Histograms'''
    assert len(dfs) == 2
    plt.xlabel("ROI") 
    plt.ylabel("Frequency")
    plt.suptitle("today:%s, h:%s, scenario:%s"%(today, histPeriods, n_scenario))

#     n_bin = histPeriods/2.
    n_bin = 100
    rois = []
    for idx, df in enumerate(dfs):
        series = df['adjROI']
        endIdx = series.index.get_loc(today)
        startIdx = endIdx - histPeriods + 1
        assert startIdx >= 0
        roi = series[startIdx: endIdx+1]
        rois.append(roi)
        print roi
        
        subplot = plt.subplot(2,3, idx+1)
        stats = [roi.mean(), roi.std(), spstats.skew(roi), spstats.kurtosis(roi)]
        subplot.set_title("%s real ROI\n%.4f, %.4f, %.4f, %.4f"%(symbols[idx], 
                          stats[0], stats[1], stats[2], stats[3]))
        subplot.set_xlim(-7., 7.)
        subplot.hist(roi.values, n_bin)
    
    #correlation
    coef = np.corrcoef(rois)
    subplot = plt.subplot(2,3, 3)
    subplot.set_title("target correlation: %.4f"%(coef[0, 1]))
    subplot.scatter(rois[0].values, rois[1].values)
    
    #samples
    n_rv = 2
    
    data = np.asarray(rois)
    Moms =  np.zeros((n_rv, 4))    
    Moms[:, 0] = data.mean(axis=1)
    Moms[:, 1] = data.std(axis=1)
    Moms[:, 2] = spstats.skew(data, axis=1)
    Moms[:, 3] = spstats.kurtosis(data, axis=1)
    corrMtx = np.corrcoef(data)
    outMtx = mom.HeuristicMomentMatching(Moms, corrMtx, n_scenario, verbose=True)
    print outMtx
    
    for idx in xrange(n_rv):
        roi = outMtx[idx]
        subplot = plt.subplot(2,3, 3+idx+1)
        stats = [roi.mean(), roi.std(), spstats.skew(roi), spstats.kurtosis(roi)]
        subplot.set_title("%s sample ROI\n%.4f, %.4f, %.4f, %.4f"%(symbols[idx], 
                          stats[0], stats[1], stats[2], stats[3]))
        subplot.set_xlim(-7., 7.)
        subplot.hist(roi, n_bin, color="green")
    
    #correlation
    coef = np.corrcoef(outMtx)
    subplot = plt.subplot(2,3, 6)
    subplot.set_title("sample correlation: %.4f"%(coef[0, 1]))
    subplot.scatter(outMtx[0], outMtx[1], color="green")
    
    
    
    plt.show()
    



if __name__ == '__main__':
    dfs = readDF(symbols)
    plotDistribution(dfs, date(2013,1,9), 10, 10)
    