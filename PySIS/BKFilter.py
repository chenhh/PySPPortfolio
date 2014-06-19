# -*- coding: utf-8 -*-
'''
Baxter-king filter
statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.filters.bkfilter.html
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


def runBKFilter():
    symbol = ('TAIEX',)
    data = pd.read_pickle(os.path.join(PySPPortfolioDir,'pkl', 
                                       'BasicFeatures', '%s.pkl'%symbol))
    startDate, endDate = date(2005,1,1), date(2013,12,31)
    rec = data[startDate:endDate]
    low, high, K = 6, 32, 12
    cycle = tsa_filter.bkfilter(rec.closePrice, low, high ,K)
    
    
    plt.plot(rec.closePrice, label="TAIEX close price")
    plt.plot(cycle, label="cycle")
    plt.title("TAIEX %s-%s, BK filter (low, high, k):(%s,%s,%s)"%(startDate, endDate, low, high, K))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    runBKFilter()