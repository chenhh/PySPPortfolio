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
from scenario.CMoment import HeuristicMomentMatching
from riskOpt.MinCVaRPortfolioSP import MinCVaRPortfolioSP

PklBasicFeaturesDir = os.path.join(os.getcwd(),'pkl', 'BasicFeatures')
symbols = ['2330', '1216','2002']
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
    outMtx = HeuristicMomentMatching(Moms, corrMtx, n_scenario, verbose=True)
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
    

def plotCVaR(symbols, today=date(2013,1,9), alpha=0.95, 
             n_histPeriod = 100, n_scenario=200):
    
    dfs = readDF(symbols)


 
    rois = []
    
    for df in dfs:
        series = df['adjROI']/100.
        endIdx = series.index.get_loc(today)
        startIdx = endIdx - n_histPeriod + 1
        assert startIdx >= 0
        roi = series[startIdx: endIdx+1]
        rois.append(roi)
        
    #samples
    n_rv = len(symbols)
    data = np.asarray(rois)
    Moms =  np.zeros((n_rv, 4))    
    Moms[:, 0] = data.mean(axis=1)
    Moms[:, 1] = data.std(axis=1)
    Moms[:, 2] = spstats.skew(data, axis=1)
    Moms[:, 3] = spstats.kurtosis(data, axis=1)
    corrMtx = np.corrcoef(data)
    outMtx = HeuristicMomentMatching(Moms, corrMtx, n_scenario, verbose=False)
    
#     print "scen:", outMtx
#     print "riskyRet:", data[:, -1]
    results = MinCVaRPortfolioSP(symbols, 
                        riskyRet=data[:, -1], 
                        riskFreeRet=0, 
                        allocatedWealth = np.zeros(n_rv),
                        depositWealth=1e6, 
                        buyTransFee=np.ones(n_rv)*0.001425, 
                        sellTransFee=np.ones(n_rv)*0.004425, 
                        alpha=alpha,
                        predictRiskyRet=outMtx, 
                        predictRiskFreeRet=0, 
                        n_scenario=n_scenario, 
                        probs=None, solver="cplex")
    print results
    weights = results['buys']
    dist = ((outMtx+1) * weights[:, np.newaxis]).sum(axis=0)
    dist.sort()
    
    quantile = int(np.ceil(n_scenario * (1-alpha)))
    print "VaR:", dist[quantile-1]
    print "CVaR:", dist[:quantile-1].mean()
   
    n_bin = 50
    plt.title("CVaR alpha:%s, scenario:%s"%(alpha, n_scenario))
    plt.hist(dist, n_bin, color="green")
    plt.show()

if __name__ == '__main__':
#     dfs = readDF(symbols)
#     plotDistribution(dfs, date(2013,1,9), 10, 10)
    plotCVaR(symbols, today=date(2013,1,10), alpha=0.95, 
             n_histPeriod = 100, n_scenario=200)