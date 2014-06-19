# -*- coding: utf-8 -*-
'''
Baxter-king filter
Christiano Fitzgerald asymmetric, random walk filter
@author: Hung-Hsin Chen
'''
import sys
import os
from datetime import date

PySPPortfolioDir = os.path.join(os.path.abspath(os.path.curdir), '..', 'PySPPortfolio')
sys.path.insert(0, PySPPortfolioDir)

import statsmodels.tsa.filters as tsa_filter
import pandas as pd
import matplotlib.pyplot as plt

from PySPPortfolio import (PklBasicFeaturesDir,)


def runCFFilter():
    symbol = ('TAIEX',)
    data = pd.read_pickle(os.path.join(PySPPortfolioDir,'pkl', 
                                       'BasicFeatures', '%s.pkl'%symbol))
    startDate, endDate = date(2005,1,1), date(2013,12,31)
    rec = data[startDate:endDate]
    low=6
    high=32
    drift=True
    cycle, trend = tsa_filter.cffilter(rec.closePrice, low, high , drift)
    
    
    plt.plot(rec.closePrice, label="TAIEX close price")
    plt.plot(cycle, label="cycle")
    plt.plot(trend, label="trend")
    plt.title("TAIEX %s-%s, CF filter (low, high, drifht):(%s,%s,%s)"%(startDate, endDate, low, high, drift))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    runCFFilter()