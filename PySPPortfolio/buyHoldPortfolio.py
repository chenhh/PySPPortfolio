# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
from __future__ import division
import numpy as np
import pandas as pd
import os
import sys
from datetime import date
import time
from stats import Performance
from cStringIO import StringIO

ProjectDir = os.path.join(os.path.abspath(os.path.curdir), '..')
sys.path.insert(0, ProjectDir)

from PySPPortfolio import (PklBasicFeaturesDir,  ExpResultsDir)
import statsmodels.tsa.stattools as sts
import statsmodels.stats.stattools as sss
import statsmodels.stats.diagnostic as ssd


def buyHoldPortfolio(symbols, startDate=date(2005,1,3), endDate=date(2013,12,31),  
                     money=1e6, buyTransFee=0.001425, sellTransFee=0.004425,
                        save_pkl=False, save_csv=True, debug=False):
    t = time.time()
    
    #read df
    dfs = []
    transDates = None
    for symbol in symbols:
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        tmp = df[startDate: endDate]
        startIdx = df.index.get_loc(tmp.index[0])
        endIdx =  df.index.get_loc(tmp.index[-1])
     
        data = df[startIdx: endIdx+1]['adjROI']/100.

        #check all data have the same transDates
        if transDates is None:
            transDates = data.index.values
        if not np.all(transDates == data.index.values):
            raise ValueError('symbol %s do not have the same trans. dates'%(symbol))
        dfs.append(data)
    
     
    #initialize
    n_rv = len(dfs)
    symbols.append('deposit')
    wealthProcess = pd.DataFrame(columns=symbols, index=transDates)
    
    #allocation
    for symbol in symbols[:-1]:
        wealthProcess[symbol][transDates[0]] = money/n_rv * (1-buyTransFee)
    wealthProcess['deposit'] = 0
    
    #buy and hold
    for sdx, symbol in enumerate(symbols[:-1]):
        for tdx, transDate in enumerate(transDates[1:]):
            tm1 = transDates[tdx]
            roi = dfs[sdx][transDate]
            wealthProcess[symbol][transDate] = wealthProcess[symbol][tm1] * (1+roi) 
    
    #sell in the last period
    for symbol in symbols[:-1]:
        wealthProcess[symbol][-1] *= (1-sellTransFee)
    
    wealth = wealthProcess.sum(axis=1)
    pROI = (wealth[-1]/1e6 -1) * 100
    prois = wealth.pct_change() * 100
    prois[0] = 0
    
    ret = sss.jarque_bera(prois)
    JB = ret[1]
        
    ret2 = sts.adfuller(prois)
    ADF = ret2[1]
    
      
    fileName = os.path.join(ExpResultsDir, 'BuyandHold_result_2005.csv')
    avgIO = StringIO()
    if not os.path.exists(fileName):
        avgIO.write('n_rv, wealth, wROI(%), ROI(%%), ROI-std, JB, ADF,')
        avgIO.write('Sharpe(%%), SortinoFull(%%), SortinoPartial(%%), downDevFull, downDevPartial\n')
        
    sharpe = Performance.Sharpe(prois)
    sortinof, ddf = Performance.SortinoFull(prois)
    sortinop, ddp = Performance.SortinoPartial(prois)
    
    avgIO.write('%s,%s,%s,%s,%s,%s,%s,'%(n_rv, wealth[-1], pROI, 
                                        prois.mean(), prois.std(),
                                        JB, ADF))
    avgIO.write('%s,%s,%s,%s,%s\n'%(sharpe*100, sortinof*100,sortinop*100, ddf*100, ddp*100))
    
    with open(fileName, 'ab') as fout:
        fout.write(avgIO.getvalue())
    avgIO.close()
  
    print "buyhold portfolio %s %s_%s pROI:%.3f%%, %.3f secs"%(startDate, endDate, n_rv,
                                                           pROI, time.time() -t )

if __name__ == '__main__':
    n_stocks = [5,10, 15, 20, 25, 30, 35, 40 , 45, 50]
#     n_stocks = [5, ]
    #20050103
    symbols = [
                '2330', '2412', '2882', '6505', '2317',
                '2303', '2002', '1303', '1326', '1301',
                '2881', '2886', '2409', '2891', '2357',
                '2382', '3045', '2883', '2454', '2880',
                '2892', '4904', '2887', '2353', '2324',
                '2801', '1402', '2311', '2475', '2888',
                '2408', '2308', '2301', '2352', '2603',
                '2884', '2890', '2609', '9904', '2610',
                '1216', '1101', '2325', '2344', '2323',
                '2371', '2204', '1605', '2615', '2201',
                ]
   
    startDate=date(2005,1,3)
    endDate=date(2013,12,31)
    for n_stock in n_stocks:
        buyHoldPortfolio(symbols[:n_stock], startDate, endDate)