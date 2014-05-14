# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
from fixedSymbolSPPortfolio import constructModelMtx
from datetime import date
import pandas as pd

import sys
import os
ProjectDir = os.path.join(os.path.abspath(os.path.curdir), '..')
sys.path.insert(0, ProjectDir)

from PySPPortfolio import (PklBasicFeaturesDir,  ExpResultsDir)


def testConstructModelMtx():
    symbols=('2330','2002') 
    startDate=date(2005,1,1)
    endDate=date(2005,1,10) 
    hist_period=3
    debug=False

    results = constructModelMtx(symbols, startDate, endDate, 
                      hist_period=hist_period, debug=debug)
#     print results
    print results['allRiskyRetMtx']
    print "T:", results['T']
    print "transDates:", results['transDates']
    print  "fullTransDates:", results[ "fullTransDates"]
#     print results['buyTransFeeMtx']
#     print results['sellTransFeeMtx']
    
    for symbol in symbols:
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        tmp = df[startDate: endDate]
        startIdx = df.index.get_loc(tmp.index[0])
        endIdx =  df.index.get_loc(tmp.index[-1])
      
  
        #index from [0, hist_period-1] for estimating statistics
        data = df[startIdx-hist_period+1: endIdx+1]['adjROI']/100.0
        print data
    
if __name__ == '__main__':
    testConstructModelMtx()