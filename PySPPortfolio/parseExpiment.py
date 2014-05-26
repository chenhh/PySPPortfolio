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
import pandas as pd
from datetime import date
import scipy.stats as spstats
ProjectDir = os.path.join(os.path.abspath(os.path.curdir), '..')
sys.path.insert(0, ProjectDir)
from stats import Performance
from PySPPortfolio import  (PklBasicFeaturesDir,  ExpResultsDir)
from time import time    
from stats import Performance
import statsmodels.tsa.stattools as sts
import statsmodels.stats.stattools as sss
    

def VaRBackTest(wealthCSV, riskCSV):
    '''
    @wealth, wealth process
    @CVaR, CVaR process
    '''
    wealth_df = pd.read_csv(wealthCSV, index_col=0, parse_dates=True)
    risk_df = pd.read_csv(riskCSV, index_col=0, parse_dates=True)

    wealth =  wealth_df.sum(axis=1)[:-1]
    CVaR = risk_df['CVaR']
    VaR = risk_df['VaR']
    
    CVaRCount = (wealth <= CVaR).sum()
    VaRCount = (wealth <= VaR).sum()
    CVaRFailRate = float(CVaRCount)/len(CVaR)
    VaRFailRate = float(VaRCount)/len(VaR)
    return CVaRFailRate, VaRFailRate 


def readWealthCSV():
    df = pd.read_csv('test_wealthProcess.csv', index_col=0, parse_dates=True)
    return df
 


def parseFixedSymbolResults():
    n_rvs = range(5, 55, 5)
    hist_periods = range(70, 130, 10)
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95", "0.99")
    global ExpResultsDir
    
    myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", "LargestMarketValue_200501")
    for n_rv in n_rvs:
        t = time()
        avgIO = StringIO()        
        avgIO.write('run, n_stock-n_rv, hist_period, alpha, time, wealth, wealth-std, wROI(%),wROI-std, JB, ADF,' )
        avgIO.write('meanROI(%%), meanROI-std, Sharpe(%%), Sharpe-std, SortinoFull(%%), SortinoFull-std,')
        avgIO.write('SortinoPartial(%%), SortinoPartial-std, downDevFull, downDevPartial, CVaRfailRate, VaRfailRate, scen err\n')
        
        for period in hist_periods:
            for alpha in alphas:
                dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                wealths, rois, elapsed, scenerr = [], [], [], []
                sharpe, sortinof, sortinop, dROI = [], [], [], []
                CVaRFailRates, VaRFailRates = [], []
                downDevF, downDevP = [], []
                JBs, ADFs = [], []
                
                for exp in exps:
                    summaryFile = os.path.join(exp, "summary.json")
                    if not os.path.exists(summaryFile):
                        continue
                    summary = json.load(open(summaryFile))                 
                    wealth = float(summary['final_wealth'])
                    print dirName, wealth
                    wealths.append(wealth)
                    rois.append((wealth/1e6-1) * 100.0)
                    elapsed.append(float(summary['elapsed']))
                    scenerr.append(summary['scen_err_cnt'])
                    try:
                        sharpe.append(float(summary['wealth_ROI_Sharpe'])*100)
                        sortinof.append(float(summary['wealth_ROI_SortinoFull'])*100)
                        sortinop.append(float(summary['wealth_ROI_SortinoPartial'])*100)
                        dROI.append((float(summary['wealth_ROI_mean']))*100)
                        downDevF.append((float(summary['wealth_ROI_downDevFull']))*100)
                        downDevP.append((float(summary['wealth_ROI_downDevPartial']))*100)
                        JBs.append(float(summary['wealth_ROI_JBTest']))
                        ADFs.append(float(summary['wealth_ROI_ADFTest']))
                        
                    except (KeyError, TypeError):
                        #read wealth process
                        csvfile = os.path.join(exp, 'wealthProcess.csv')
                        df = pd.read_csv( csvfile, index_col=0, parse_dates=True)
                        proc = df.sum(axis=1)
                        wrois =  proc.pct_change()
                        wrois[0] = 0
                        
                        dROIval = wrois.mean()
                        dROI.append(dROIval*100)
                        sharpeVal = Performance.Sharpe(wrois)
                        sortinofVal, ddf = Performance.SortinoFull(wrois)
                        sortinopVal, ddp = Performance.SortinoPartial(wrois)
                        ret = sss.jarque_bera(wrois)
                        JB = ret[1]
        
                        ret2 = sts.adfuller(wrois)
                        ADF = ret2[1]
                        
                        sharpe.append(sharpeVal*100)
                        sortinof.append(sortinofVal*100)
                        sortinop.append(sortinopVal*100)
                        downDevF.append(ddf*100)
                        downDevP.append(ddp*100)
                        JBs.append(JB)
                        ADFs.append(ADF)
                     
                        
                        summary['wealth_ROI_mean'] = dROIval
                        summary['wealth_ROI_Sharpe'] = sharpeVal 
                        summary['wealth_ROI_SortinoFull'] = sortinofVal
                        summary['wealth_ROI_SortinoPartial'] = sortinopVal
                        summary['wealth_ROI_downDevFull'] = ddf
                        summary['wealth_ROI_downDevPartial'] = ddp
                        summary['wealth_ROI_JBTest'] = JB
                        summary['wealth_ROI_ADFTest'] = ADF
                        
                        fileName = os.path.join(exp, 'summary.json')
                        with open (fileName, 'w') as fout:
                            json.dump(summary, fout, indent=4)
                     
                    try:
                        CVaRFailRate = float(summary['CVaR_failRate']*100)
                        VaRFailRate = float(summary['VaR_failRate']*100)
                        CVaRFailRates.append(CVaRFailRate)
                        VaRFailRates.append(VaRFailRate)
                        
                    except (KeyError,TypeError):
                        wealthFile = os.path.join(exp, 'wealthProcess.csv')
                        riskFile = os.path.join(exp, 'riskProcess.csv')
                        CVaRFailRate, VaRFailRate = VaRBackTest(wealthFile, riskFile)
                        print "CVaR fail:%s, VaR fail:%s"%(CVaRFailRate, VaRFailRate)
                        summary['CVaR_failRate'] = CVaRFailRate
                        summary['VaR_failRate'] = VaRFailRate
                        
                        CVaRFailRates.append(CVaRFailRate*100)
                        VaRFailRates.append(VaRFailRate*100)
                        
#                         fileName = os.path.join(exp, 'summary.json')
#                         with open (fileName, 'w') as fout:
#                             json.dump(summary, fout, indent=4)
#                         
                    
                        
                rois = np.asarray(rois)
                wealths = np.asarray(wealths)
                elapsed = np.asarray(elapsed)
                scenerr = np.asarray(scenerr)
                sharpe = np.asarray(sharpe)
                sortinof = np.asarray(sortinof) 
                sortinop = np.asarray(sortinop)
                dROI =  np.asarray(dROI) 
                CVaRFailRates = np.asarray(CVaRFailRates)
                VaRFailRates = np.asarray(VaRFailRates)
                downDevF = np.asarray(downDevF)
                downDevP = np.asarray(downDevP)
                JBs = np.asarray(JBs)
                ADFs = np.asarray(ADFs)
                                
                avgIO.write("%s, %s-%s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %.2f\n"%(
                                len(rois), n_rv, n_rv, period, alpha,  elapsed.mean(),
                                wealths.mean(), wealths.std(), 
                                rois.mean(), rois.std(), max(JBs), max(ADFs),
                                dROI.mean(), dROI.std(), 
                                sharpe.mean(), sharpe.std(), 
                                sortinof.mean(), sortinof.std(), 
                                sortinop.mean(), sortinop.std(),
                                downDevF.mean(), downDevP.mean(),
                                CVaRFailRates.mean(), VaRFailRates.mean(),
                                scenerr.mean() ))
                
        resFile =  os.path.join(ExpResultsDir, 'avg_fixedSymbolSPPortfolio_n%s_result_2005.csv'%(n_rv))
        with open(resFile, 'wb') as fout:
            fout.write(avgIO.getvalue())
        avgIO.close()
        print "n_rv:%s OK, elapsed %.3f secs"%(n_rv, time()-t)


def parseDynamicSymbolResults(n_rv=50):

    n_stocks =  range(5, 55, 5)
    hist_periods = range(70, 130, 10)
    n_scenario = 200
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95")
    global ExpResultsDir
    
    myDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", 
                         "LargestMarketValue_200501_rv%s"%(n_rv))
    
    for n_stock in n_stocks:
        t = time()
        avgIO = StringIO()        
        avgIO.write('run, n_stock ,n_rv, hist_period, alpha, runtime, wealth, wROI(%), dROI(%%), Sharpe(%%), SortinoFull(%%), SortinoPartial(%%), scen err\n')
        
        for period in hist_periods:
            for alpha in alphas:
                dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_stock, period, alpha)
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                
                #multiple runs of a parameter
                wealths, rois, elapsed, scenerr = [], [], [], []
                sharpe, sortinof, sortinop, dROI = [], [], [], []
                for exp in exps:
                    print exp
                    summary = json.load(open(os.path.join(exp, "summary.json")))
                    wealth = float(summary['final_wealth'])
                    wealths.append(wealth)
                    rois.append((wealth/1e6-1) * 100.0)
                    elapsed.append(float(summary['elapsed']))
                    scenerr.append(summary['scen_err_cnt'])
                    try:
                        sharpe.append(float(summary['wealth_ROI_Sharpe'])*100)
                        sortinof.append(float(summary['wealth_ROI_SortinoFull'])*100)
                        sortinop.append(float(summary['wealth_ROI_SortinoPartial'])*100)
                        dROI.append((float(summary['wealth_ROI_mean']))*100)
                    except KeyError:
                        #read wealth process
                        csvfile = os.path.join(exp, 'wealthProcess.csv')
                        df = pd.read_csv( csvfile, index_col=0, parse_dates=True)
                        proc = df.sum(axis=1)
                        wrois =  proc.pct_change()
                        wrois[0] = 0
                            
                        dROI.append(wrois.mean()*100)
                        sharpe.append(Performance.Sharpe(wrois)*100)
                        sortinof.append(Performance.SortinoFull(wrois)*100)
                        sortinop.append(Performance.SortinoPartial(wrois)*100)
                    
                
                rois = np.asarray(rois)
                wealths = np.asarray(wealths)
                elapsed = np.asarray(elapsed)
                scenerr = np.asarray(scenerr)    
                sharpe = np.asarray(sharpe)
                sortinof = np.asarray(sortinof) 
                sortinop = np.asarray(sortinop)
                dROI =  np.asarray(dROI) 
                
                avgIO.write("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n"%(
                                len(rois), n_stock, n_rv, period, alpha,  elapsed.mean(),
                                wealths.mean(), rois.mean(), dROI.mean(), sharpe.mean(), sortinof.mean(), sortinop.mean(),
                                scenerr.mean() ))
        
        #write results    
        resFile =  os.path.join(ExpResultsDir, 
                        'avg_dynamicSymbolSPPortfolio_rv%s_n%s_result_2005.csv'%(n_rv, n_stock))
        with open(resFile, 'wb') as fout:
            fout.write(avgIO.getvalue())
        avgIO.close()
        print "n_stock:%s OK, elapsed %.3f secs"%(n_stock, time()-t)


def parseWCVaRSymbolResults():
    n_rvs = range(5, 55, 5)
    hist_periods = range(70, 130, 10)
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95")
    global ExpResultsDir
    
    myDir = os.path.join(ExpResultsDir, "fixedSymbolWCVaRSPPortfolio", "LargestMarketValue_200501")
    for n_rv in n_rvs:
        t = time()
        avgIO = StringIO()        
        avgIO.write('run, n_stock ,n_rv, hist_period, alpha, runtime, wealth, wROI(%), dROI(%%), Sharpe(%%), SortinoFull(%%), SortinoPartial(%%), scen err\n')
        
        for alpha in alphas:
            dirName = "fixedSymbolWCVaRSPPortfolio_n%s_p70-80-90-100-110-120_s100_a%s"%(n_rv, alpha)
            exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
            wealths, rois, elapsed, scenerr = [], [], [], []
            sharpe, sortinof, sortinop, dROI = [], [], [], []
            for exp in exps:
                summary = json.load(open(os.path.join(exp, "summary.json")))
                wealth = float(summary['final_wealth'])
                print dirName, wealth
                wealths.append(wealth)
                rois.append((wealth/1e6-1) * 100.0)
                elapsed.append(float(summary['elapsed']))
                scenerr.append(summary['scen_err_cnt'])
                try:
                    sharpe.append(float(summary['wealth_ROI_Sharpe'])*100)
                    sortinof.append(float(summary['wealth_ROI_SortinoFull'])*100)
                    sortinop.append(float(summary['wealth_ROI_SortinoPartial'])*100)
                    dROI.append((float(summary['wealth_ROI_mean']))*100)
                except KeyError:
                    #read wealth process
                    csvfile = os.path.join(exp, 'wealthProcess.csv')
                    df = pd.read_csv( csvfile, index_col=0, parse_dates=True)
                    proc = df.sum(axis=1)
                    wrois =  proc.pct_change()
                    wrois[0] = 0
                        
                    dROI.append(wrois.mean()*100)
                    sharpe.append(Performance.Sharpe(wrois)*100)
                    sortinof.append(Performance.SortinoFull(wrois)*100)
                    sortinop.append(Performance.SortinoPartial(wrois)*100)
                    

            rois = np.asarray(rois)
            wealths = np.asarray(wealths)
            elapsed = np.asarray(elapsed)
            scenerr = np.asarray(scenerr)
            sharpe = np.asarray(sharpe)
            sortinof = np.asarray(sortinof) 
            sortinop = np.asarray(sortinop)
            dROI =  np.asarray(dROI) 
            
            avgIO.write("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n"%(
                            len(rois), n_rv, n_rv, "70-80-90-100-110-120", alpha,  elapsed.mean(),
                            wealths.mean(), rois.mean(), dROI.mean(), sharpe.mean(), sortinof.mean(), sortinop.mean(),
                            scenerr.mean() ))
                
        resFile =  os.path.join(ExpResultsDir, 'avg_fixedWCVaRSPPortfolio_n%s_result_2005.csv'%(n_rv))
        with open(resFile, 'wb') as fout:
            fout.write(avgIO.getvalue())
        avgIO.close()
        print "n_rv:%s OK, elapsed %.3f secs"%(n_rv, time()-t)




def individualSymbolStats():
  
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
    
    statIO = StringIO()        
    statIO.write('rank & symbol & mean & stdev. & skewness & kurtosis & JB & ADF & $R_{CUM}$ & $R_{AR}$ \\\ \hline \n')
    
    for idx, symbol in enumerate(symbols):
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        tmp = df[startDate: endDate]
        rois = tmp['adjROI'].values

        mean = rois.mean()
        std = rois.std()
        skew = spstats.skew(rois)
        kurt = spstats.kurtosis(rois)
#         sharpe = Performance.Sharpe(rois)
#         sortinof = Performance.SortinoFull(rois)
#         sortinop = Performance.SortinoPartial(rois)
        print rois
#         k2, pval = spstats.normaltest(rois)
        ret = sss.jarque_bera(rois)
        JB = ret[1]
        
        ret2 = sts.adfuller(rois)
        ADF = ret2[1]
        
        rtmp = rois/100 + 1
        rtmp[1] -= 0.001425 #buy fee
        rtmp[-1] -= 0.004425 #sell fee
        R_cum = rtmp[1:].prod() - 1 
        AR_cum = np.power((R_cum+1), 1./9) -1  
        
        statIO.write('%2d & %s & %8.4f & %8.4f & %8.4f & %8.4f & %8.4e & %8.4e & %8.4f & %8.4f   \\\ \hline \n'%(
                        idx+1, symbol, mean, std, skew, kurt,  JB, ADF, R_cum*100, AR_cum*100))
        print symbol, R_cum, AR_cum
    
    resFile =  os.path.join(ExpResultsDir, 'symbol_daily_stats.txt')
    with open(resFile, 'wb') as fout:
        fout.write(statIO.getvalue())
        statIO.close()
    
    statIO.close()
     

def groupSymbolStats():
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

    statIO = StringIO()        
    statIO.write(' $n$ & mean & stdev. & skewness & kurtosis & JB & ADF & $R_{CUM}$ & $R_{AR}$ \\\ \hline \n')
    
    grois = []
    for idx, symbol in enumerate(symbols):
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        tmp = df[startDate: endDate]
        rois = tmp['adjROI'].values
        grois.append(rois)
    grois = np.asarray(grois)
    
    print grois.shape
    
    for psize in range(5, 50+5, 5):
        rois = grois[:psize, :].mean(axis=0)
        mean = rois.mean()
        std = rois.std()
        skew = spstats.skew(rois)
        kurt = spstats.kurtosis(rois)

#         k2, pval = spstats.normaltest(rois)
        ret = sss.jarque_bera(rois)
        JB = ret[1]
        
        ret2 = sts.adfuller(rois)
        ADF = ret2[1]

        rtmp = rois/100 + 1
        rtmp[1] -= 0.001425 #buy fee
        rtmp[-1] -= 0.004425 #sell fee
        R_cum = rtmp[1:].prod() - 1 
        AR_cum = np.power((R_cum+1), 1./9) -1  
     
        statIO.write('%2d  & %8.4f & %8.4f & %8.4f & %8.4f & %8.4e & %8.4e  & %8.4f & %8.4f  \\\ \hline \n'%(
                    psize, mean, std, skew, kurt,  JB, ADF, R_cum*100, AR_cum*100))
        print psize, R_cum, AR_cum
     
    resFile =  os.path.join(ExpResultsDir, 'group_symbol_daily_stats.txt')
    with open(resFile, 'wb') as fout:
        fout.write(statIO.getvalue())
        statIO.close()
     
    statIO.close()


def comparisonStats():
    symbols = [
        'TAIEX', '0050',
    ]
    
    startDate=date(2005,1,3)
    endDate=date(2013,12,31)
    
    statIO = StringIO()        
    statIO.write('symbol & mean & stdev. & skewness & kurtosis & JB & ADF & $R_{CUM}$ & $R_{AR}$ \\\ \hline \n')
    
    for idx, symbol in enumerate(symbols):
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        print symbol, df.columns
        tmp = df[startDate: endDate]
        rois = tmp['adjROI'].values

        mean = rois.mean()
        std = rois.std()
        skew = spstats.skew(rois)
        kurt = spstats.kurtosis(rois)
        print rois
#         k2, pval = spstats.normaltest(rois)
        
        ret = sss.jarque_bera(rois)
        JB = ret[1]
        
        ret2 = sts.adfuller(rois)
        ADF = ret2[1]

        
        rtmp = rois/100 + 1
        rtmp[1] -= 0.001425 #buy fee
        rtmp[-1] -= 0.004425 #sell fee
        R_cum = rtmp[1:].prod() - 1 
        AR_cum = np.power((R_cum+1), 1./9) -1  
        
        statIO.write('%s & %8.4f & %8.4f & %8.4f & %8.4f & %8.4e & %8.4e & %8.4f & %8.4f  \\\ \hline \n'%(
                        symbol, mean, std, skew, kurt,  JB, ADF, R_cum*100, AR_cum*100))
        print symbol, R_cum, AR_cum
    
    resFile =  os.path.join(ExpResultsDir, 'comparison_daily_stats.txt')
    with open(resFile, 'wb') as fout:
        fout.write(statIO.getvalue())
        statIO.close()
    
    statIO.close()

def csv2Pkl():
    n_rvs = range(5, 55, 5)
    hist_periods = range(70, 130, 10)
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95", "0.99")
    global ExpResultsDir
    
    myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", "LargestMarketValue_200501")
    for n_rv in n_rvs:
        for period in hist_periods:
            for alpha in alphas:
                dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                
                for exp in exps:
                    
                    for fileName in ['wealthProcess.csv', 'riskProcess.csv', 'actionProcess.csv']:
                        csvFile =os.path.join(exp, fileName)
                        df = pd.read_csv(csvFile, index_col=0, parse_dates=True)
                        procName = csvFile[csvFile.rfind('/')+1:csvFile.rfind('.')]
                        dfFile = os.path.join(exp, "%s.pkl"%(procName))
                        df.save(dfFile)
                        
                        #if transform successful
                        if os.path.exists(csvFile) and os.path.exists(dfFile):
                            os.remove(csvFile) 
                    
                    print exp

if __name__ == '__main__':
#     readWealthCSV()
#     parseFixedSymbolResults()
#     parseDynamicSymbolResults()
#     parseWCVaRSymbolResults()
#     individualSymbolStats()
#     groupSymbolStats()
#     comparisonStats()
    csv2Pkl()