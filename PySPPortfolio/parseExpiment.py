# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

'''

import simplejson as json 
from cStringIO import StringIO
import numpy as np
import sys
import os
from glob import glob
ProjectDir = os.path.join(os.path.abspath(os.path.curdir), '..')
sys.path.insert(0, ProjectDir)

from PySPPortfolio import ExpResultsDir
print ExpResultsDir
from time import time    


def parseFixedSymbolResults():
    n_rvs = (5, 10)
    hist_periods = range(10, 130, 10)
    n_scenario = 200
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95", "0.99")
    global ExpResultsDir
    
    myDir = os.path.join(ExpResultsDir, "LargestMarketValue_200501")
    for n_rv in n_rvs:
        t = time()
        avgIO = StringIO()        
        avgIO.write('run, n_rv, hist_period, alpha, runtime, wealth, wstd, ROI(%), rstd, scen err\n')
        
        for period in hist_periods:
            for alpha in alphas:
                dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                wealths, rois, elapsed, scenerr = [], [], [], []
                for exp in exps:
                    print exp
                    summary = json.load(open(os.path.join(exp, "summary.json")))
                    wealth = float(summary['final_wealth'])
                    wealths.append(wealth)
                    rois.append((wealth/1e6-1) * 100.0)
                    elapsed.append(float(summary['elapsed']))
                    scenerr.append(summary['scen_err_cnt'])
                
                rois = np.asarray(rois)
                wealths = np.asarray(wealths)
                elapsed = np.asarray(elapsed)
                scenerr = np.asarray(scenerr)
                avgIO.write("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n"%(
                                len(rois), n_rv, period, alpha,  elapsed.mean(),
                                wealths.mean(), wealths.std(), 
                                rois.mean(), rois.std(), scenerr.mean() ))
                
        resFile =  os.path.join(ExpResultsDir, 'avg_n%s_result_2005.csv'%(n_rv))
        with open(resFile, 'wb') as fout:
            fout.write(avgIO.getvalue())
        avgIO.close()
        print "n_rv:%s OK, elapsed %.3f secs"%(n_rv, time()-t)


def parseDynamicSymbolResults(n_rv=50):
    n_stocks = (5, 10)
    hist_periods = range(10, 130, 10)
    n_scenario = 200
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95", "0.99")
    global ExpResultsDir
    
    myDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", "LargestMarketValue_200501_rv%s"%(n_rv))
    for n_stock in n_stocks:
        t = time()
        avgIO = StringIO()        
        avgIO.write('run, n_stock ,n_rv, hist_period, alpha, runtime, wealth, wROI, Sharpe, SortinoFull, SortinoPartial, ROI(%), scen err\n')
        
        for period in hist_periods:
            for alpha in alphas:
                dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_stock, period, alpha)
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                
                #multiple runs of a parameter
                wealths, rois, elapsed, scenerr = [], [], [], []
                for exp in exps:
                    print exp
                    summary = json.load(open(os.path.join(exp, "summary.json")))
                    wealth = float(summary['final_wealth'])
                    wealths.append(wealth)
                    rois.append((wealth/1e6-1) * 100.0)
                    elapsed.append(float(summary['elapsed']))
                    scenerr.append(summary['scen_err_cnt'])
                
                rois = np.asarray(rois)
                wealths = np.asarray(wealths)
                elapsed = np.asarray(elapsed)
                scenerr = np.asarray(scenerr)
                avgIO.write("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n"%(
                                len(rois), n_rv, period, alpha,  elapsed.mean(),
                                wealths.mean(), wealths.std(), 
                                rois.mean(), rois.std(), scenerr.mean() ))
                
        resFile =  os.path.join(ExpResultsDir, 'avg_n%s_result_2005.csv'%(n_rv))
        with open(resFile, 'wb') as fout:
            fout.write(avgIO.getvalue())
        avgIO.close()
        print "n_rv:%s OK, elapsed %.3f secs"%(n_rv, time()-t)

if __name__ == '__main__':
    parseResults()
   