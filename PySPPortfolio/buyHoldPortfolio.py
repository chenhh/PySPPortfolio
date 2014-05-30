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
import scipy.stats as spstats
from cStringIO import StringIO
import scipy.stats as spstats

ProjectDir = os.path.join(os.path.abspath(os.path.curdir), '..')
sys.path.insert(0, ProjectDir)

from PySPPortfolio import (PklBasicFeaturesDir,  ExpResultsDir)
import statsmodels.tsa.stattools as sts
import statsmodels.stats.stattools as sss
import statsmodels.stats.diagnostic as ssd


def buyHoldPortfolio(symbols, startDate=date(2005,1,3), endDate=date(2013,12,31),  
                     money=1e6, buyTransFee=0.001425, sellTransFee=0.004425,
                        save_latex=False, save_csv=True, debug=False):
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
    prois = wealth.pct_change()
    prois[0] = 0
    
    ret = sss.jarque_bera(prois)
    JB = ret[1]
        
    ret2 = sts.adfuller(prois)
    ADF = ret2[1]
    
    
    resultDir = os.path.join(ExpResultsDir, "BuyandHoldPortfolio")
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)
    
    fileName = os.path.join(resultDir, 'BuyandHold_result_2005.csv')
    statName = os.path.join(resultDir, 'BuyandHold_result_2005.txt')
    
    df_name = os.path.join(resultDir,"wealthProcess_n%s.pkl"%(len(dfs)))
    df2_name = os.path.join(resultDir,"wealthSum_n%s.pkl"%(len(dfs)))
    csv_name = os.path.join(resultDir,"wealthProcess_n%s.csv"%(len(dfs)))
    csv2_name = os.path.join(resultDir,"wealthSum_n%s.csv"%(len(dfs)))
    wealthProcess.to_csv(csv_name)
    wealth.to_csv(csv2_name)
    wealthProcess.to_pickle(df_name)
    wealth.to_pickle(df2_name)
    
    csvIO = StringIO()
    statIO = StringIO()
    if not os.path.exists(fileName):

        csvIO.write('n_rv, wealth, wROI(%), ROI(%%), ROI-std, skew, kurt, JB, ADF,')
        csvIO.write('Sharpe(%%), SortinoFull(%%), SortinoPartial(%%), downDevFull, downDevPartial\n')
        statIO.write('$n$ & $R_{C}$(\%) & $R_{A}$(\%) & $\mu$(\%) & $\sigma$(\%) & skew & kurt & $S_p$(\%) & $S_o$(\%)  & JB & ADF \\\ \hline \n')

    sharpe = Performance.Sharpe(prois)
    sortinof, ddf = Performance.SortinoFull(prois)
    sortinop, ddp = Performance.SortinoPartial(prois)
    

    csvIO.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,'%(n_rv, wealth[-1], pROI, 
                                        prois.mean()*100, prois.std()*100,
                                        spstats.skew(prois),
                                        spstats.kurtosis(prois),
                                        JB, ADF))
    csvIO.write('%s,%s,%s,%s,%s\n'%(sharpe*100, sortinof*100,sortinop*100, ddf*100, ddp*100))
    statIO.write('%2d &  %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2e & %4.2e \\\ \hline \n'%(
                        n_rv,  pROI, (np.power(wealth[-1]/1e6, 1./9)-1)*100,  
                        prois.mean()*100, prois.std()*100, 
                        spstats.skew(prois), 
                        spstats.kurtosis(prois),
                        sharpe*100, sortinof*100,  JB, ADF ))
    
    with open(fileName, 'ab') as fout:
        fout.write(csvIO.getvalue())
    csvIO.close()
    
    with open(statName, 'ab') as fout:
        fout.write(statIO.getvalue())
    statIO.close()
  
    print "buyhold portfolio %s %s_%s pROI:%.3f%%, %.3f secs"%(startDate, endDate, n_rv,
                                                           pROI, time.time() -t )
    
    
def y2yBuyHold():
    t = time.time()
    n_rvs = range(5, 50+5, 5)
    years = range(2005, 2013+1)
    resultDir = os.path.join(ExpResultsDir, "BuyandHoldPortfolio")
    
    avgIO = StringIO()        
    avgIO.write('startDate, endDate, n_stock, wealth1, wealth2,  wROI(%), JB, ADF,' )
    avgIO.write('meanROI(%%), Sharpe(%%), SortinoFull(%%), SortinoPartial(%%),')
    avgIO.write(' downDevFull, downDevPartial\n')
    
    for n_rv in n_rvs:
        df =  pd.read_pickle(os.path.join(resultDir,"wealthSum_n%s.pkl"%(n_rv)))
        
        for year in years:
            startDate = date(year, 1, 1)
            endDate = date(year, 12, 31)
            print startDate, endDate
            wealths = df[startDate:endDate]
            wrois =  wealths.pct_change()         
            wrois[0] = 0
            
            wealth1 =  wealths[0]
            wealth2 =  wealths[-1] * (1-0.004425)
            roi = (wealth2/wealth1 - 1) 
            
            ret = sss.jarque_bera(wrois)
            JB = ret[1]
            ret2 = sts.adfuller(wrois)
            ADF = ret2[1]

            sharpe = Performance.Sharpe(wrois)
            sortinof, ddf = Performance.SortinoFull(wrois)
            sortinop, ddp = Performance.SortinoPartial(wrois)
 
            
            avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,"%( wealths.index[0].strftime("%Y-%m-%d"),
                 wealths.index[-1].strftime("%Y-%m-%d"), n_rv, wealth1, wealth2, roi*100, JB, ADF))
            avgIO.write("%s,%s,%s,%s,"%(wrois.mean()*100, sharpe*100, sortinof*100, sortinop*100))
            avgIO.write("%s,%s\n"%(ddf*100, ddp*100))
 
    resFile =  os.path.join(ExpResultsDir, 'y2yfixedBuyandHold_result_2005.csv')
    with open(resFile, 'wb') as fout:
        fout.write(avgIO.getvalue())
        avgIO.close()
    print "y2yBuyandHold OK, elapsed %.3f secs"%(time.time()-t)

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
#     y2yBuyHold()