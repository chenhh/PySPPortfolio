# -*- coding: utf-8 -*-
'''
Hodrick-Prescott filter
http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.filters.hpfilter.html
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


def runHPFilter():
    symbol = ('TAIEX',)
    data = pd.read_pickle(os.path.join(PySPPortfolioDir,'pkl', 
                                       'BasicFeatures', '%s.pkl'%symbol))
    startDate, endDate = date(2005,1,1), date(2013,12,31)
    rec = data[startDate:endDate]
    HP_lambda = 1600
    cycle, trend = tsa_filter.hpfilter(rec.closePrice, lamb=HP_lambda)
    print "cycle:", cycle
    print "trend:", trend
    
    
    plt.plot(rec.closePrice, label="TAIEX close price")
    plt.plot(cycle, label="cycle")
    plt.plot(trend, label="trend")
    plt.title("TAIEX %s-%s, HP filter lambda:%s"%(startDate, endDate, HP_lambda))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    runHPFilter()