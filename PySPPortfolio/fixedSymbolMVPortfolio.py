# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
start from 2005/1/1 to 2013/12/31
'''
from __future__ import division
import argparse
import os
import platform
import time
from datetime import date
import numpy as np
 
import scipy.stats as spstats
import pandas as pd
from stats import Performance
from riskOpt.MeanVariance import MeanVariance

import simplejson as json 
import sys
from cStringIO import StringIO
ProjectDir = os.path.join(os.path.abspath(os.path.curdir), '..')
sys.path.insert(0, ProjectDir)

from fixedSymbolSPPortfolio import constructModelMtx
from PySPPortfolio import (PklBasicFeaturesDir,  ExpResultsDir)


def fixedSymbolMVPortfolio(symbols, startDate, endDate,  money=1e6,
            hist_period=20, buyTransFee=0.001425, sellTransFee=0.004425,
            alpha=0.95, solver="cplex", save_pkl=False, save_csv=True, 
            debug=False):
    t0 = time.time()
    param = constructModelMtx(symbols, startDate, endDate, money, hist_period,
                              buyTransFee, sellTransFee, alpha, debug)
    print "constructModelMtx %.3f secs"%(time.time()-t0)
    
    n_rv, T =param['n_rv'], param['T']
    allRiskyRetMtx = param['allRiskyRetMtx']
    riskFreeRetVec = param['riskFreeRetVec']
    buyTransFeeMtx = param['buyTransFeeMtx']
    sellTransFeeMtx = param['sellTransFeeMtx']
    allocatedWealth = param['allocatedWealth']
    depositWealth = param['depositWealth']
    transDates = param['transDates']
    fullTransDates = param['fullTransDates']
    
    #process from t=0 to t=(T+1)
    buyProcess = np.zeros((n_rv, T))
    sellProcess = np.zeros((n_rv, T))
    wealthProcess = np.zeros((n_rv, T+1))
    depositProcess = np.zeros(T+1)
    VaRProcess = np.zeros(T)
    CVaRProcess = np.zeros(T)


def testFixedSymbolMVPortfolio():
    fixedSymbolMVPortfolio(symbols, startDate, endDate,  money=1e6,
            hist_period=20, buyTransFee=0.001425, sellTransFee=0.004425,
            alpha=0.95, solver="cplex", save_pkl=False, save_csv=True, 
            debug=False)
    
if __name__ == '__main__':
    testFixedSymbolMVPortfolio()