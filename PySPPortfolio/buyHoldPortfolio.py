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

ProjectDir = os.path.join(os.path.abspath(os.path.curdir), '..')
sys.path.insert(0, ProjectDir)

from PySPPortfolio import (PklBasicFeaturesDir,  ExpResultsDir)

def buyHoldPortfolio(symbols, startDate=date(2005,1,1), endDate=date(2013,12,31),  
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
     
        data = df[startIdx: endIdx+1]

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
            roi = dfs[sdx]['adjROI'][transDate]/100.0
            wealthProcess[symbol][transDate] = wealthProcess[symbol][tm1] * (1+roi) 
    
    #sell in the last period
    for symbol in symbols[:-1]:
        wealthProcess[symbol][-1] *= (1-sellTransFee)
        wealthProcess['deposit'][-1] +=  wealthProcess[symbol][-1]
        wealthProcess[symbol][-1] = 0
    
    print wealthProcess.tail(5)
    pROI = (wealthProcess['deposit'][-1]/1e6 -1) * 100
    print "buyhold portfolio %s_%s pROI:%.3f%%, %.3f secs"%(startDate, endDate, 
                                                           pROI, time.time() -t )

if __name__ == '__main__':
    n_stock = 10
    symbols = ['2330', '2317', '6505', '2412', '2454',
                '2882', '1303', '1301', '1326', '2881'
                ]
    buyHoldPortfolio(symbols[:n_stock])