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
import shutil    

def VaRBackTest(wealth_df, risk_df):
    '''
    @wealth, wealth process
    @CVaR, CVaR process
    '''
   
    if len(wealth_df.index) == len(risk_df.index)+1:
        wealth =  wealth_df.sum(axis=1)[:-1]
    elif len(wealth_df.index) == len(risk_df.index):
        wealth =  wealth_df.sum(axis=1)
        
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
#     n_rvs = [50,]
    hist_periods = range(50, 130, 10)
#     hist_periods = [60,]
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95", "0.99")
    global ExpResultsDir
    
    myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", "LargestMarketValue_200501")
    for n_rv in n_rvs:
        t = time()
        avgIO = StringIO()        
        avgIO.write('run, n_stock-n_rv, hist_period, alpha, time, wealth, wealth-std, wROI(%),wROI-std, JB, ADF,' )
        avgIO.write('meanROI(%%), stdev, skew, kurt, Sharpe(%%), Sharpe-std, SortinoFull(%%), SortinoFull-std,')
        avgIO.write('SortinoPartial(%%), SortinoPartial-std, downDevFull, downDevPartial, CVaRfailRate, VaRfailRate, scen err\n')
        
        for period in hist_periods:
            if n_rv == 50 and period == 50:
                continue
            
            for alpha in alphas:
                dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                wealths, rois, elapsed, scenerr = [], [], [], []
                sharpe, sortinof, sortinop, dROI = [], [], [], []
                skews, kurts = [], []
                CVaRFailRates, VaRFailRates = [], []
                downDevF, downDevP = [], []
                JBs, ADFs = [], []
                
                for exp in exps:
                    summaryFile = os.path.join(exp, "summary.json")
#                     if not os.path.exists(summaryFile):
#                         continue
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
                        skews.append(float(summary['wealth_ROI_skew']))
                        kurts.append(float(summary['wealth_ROI_kurt']))
                        
                    except (KeyError, TypeError):
                        #read wealth process

                        df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
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
                        
                        skew = spstats.skew(wrois)
                        kurt = spstats.kurtosistest(wrois)
                        skews.append(skew)
                        kurts.append(kurt)
                        
                        summary['wealth_ROI_mean'] = dROIval
                        summary['wealth_ROI_Sharpe'] = sharpeVal 
                        summary['wealth_ROI_SortinoFull'] = sortinofVal
                        summary['wealth_ROI_SortinoPartial'] = sortinopVal
                        summary['wealth_ROI_downDevFull'] = ddf
                        summary['wealth_ROI_downDevPartial'] = ddp
                        summary['wealth_ROI_JBTest'] = JB
                        summary['wealth_ROI_ADFTest'] = ADF
                        summary['wealth_ROI_skew'] = skew
                        summary['wealth_ROI_kurt'] = kurt
                        
                        fileName = os.path.join(exp, 'summary.json')
                        with open (fileName, 'w') as fout:
                            json.dump(summary, fout, indent=4)
                     
                    try:
                        CVaRFailRate = float(summary['CVaR_failRate']*100)
                        VaRFailRate = float(summary['VaR_failRate']*100)
                        CVaRFailRates.append(CVaRFailRate)
                        VaRFailRates.append(VaRFailRate)
                        
                    except (KeyError,TypeError):
                        wealth_df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
                        risk_df = pd.read_pickle(os.path.join(exp, 'riskProcess.pkl'))
                        
                        CVaRFailRate, VaRFailRate = VaRBackTest(wealth_df, risk_df)
                        print "CVaR fail:%s, VaR fail:%s"%(CVaRFailRate, VaRFailRate)
                        summary['CVaR_failRate'] = CVaRFailRate
                        summary['VaR_failRate'] = VaRFailRate
                        
                        CVaRFailRates.append(CVaRFailRate*100)
                        VaRFailRates.append(VaRFailRate*100)
                        
                        fileName = os.path.join(exp, 'summary.json')
                        with open (fileName, 'w') as fout:
                            json.dump(summary, fout, indent=4)
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
                skews = np.asarray(skews)
                kurts = np.asarray(kurts)
                                
                avgIO.write("%s, %s-%s, %s, %s, %s,%s,%s,%s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %.2f\n"%(
                                len(rois), n_rv, n_rv, period, alpha,  elapsed.mean(),
                                wealths.mean(), wealths.std(), 
                                rois.mean(), rois.std(), max(JBs), max(ADFs),
                                dROI.mean(), dROI.std(), skews.mean(),
                                kurts.mean(), 
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


def parseBestFixedSymbol2Latex():
    n_rvs = range(5, 55, 5)
    hist_periods = range(50, 130, 10)
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95", "0.99")
    
    global ExpResultsDir
    myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", "LargestMarketValue_200501")
    outFile = os.path.join(ExpResultsDir, "fixedSymbolBestParam.txt")
    
    for n_rv in n_rvs:
        t = time()
       
        statIO = StringIO()
        if not os.path.exists(outFile):        
            statIO.write('$n-h-\alpha$ & $R_{C}$(\%) & $R_{A}$(\%) & $\mu$(\%) & $\sigma$(\%) & skew & kurt & $S_p$(\%) & $S_o$(\%)  & JB & ADF  \\\ \hline \n')
        
        currentBestParam = {"period": 0, "alpha": 0, "wealths": 0}
        for period in hist_periods:
            for alpha in alphas:
                dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                wealths = []
           
                for exp in exps:
                    summaryFile = os.path.join(exp, "summary.json")
                    summary = json.load(open(summaryFile))                 
                    wealth = float(summary['final_wealth'])
                    wealths.append(wealth)
                wealths = np.asarray(wealths)
                
                if wealths.mean() > currentBestParam['wealths']:
                    currentBestParam['period'] = period
                    currentBestParam['alpha'] = alpha
                    currentBestParam['wealths'] = wealths.mean()
        
        #get the best param
        print "n_rv:%s bestParam p:%s a:%s"%(n_rv, currentBestParam['period'], currentBestParam['alpha'])
        dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, currentBestParam['period'], 
                                                             currentBestParam['alpha'])
        exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
        wealths, RAs,RCs = [], [], []
        dROIs, skews, kurts = [], [], []
        sharpes, sortinofs = [], []
        JBs, ADFs = [], []
        CVaRFailRates, VaRFailRates = [], []
        
        for exp in exps:
            summaryFile = os.path.join(exp, "summary.json")
            summary = json.load(open(summaryFile))                 
            wealth = float(summary['final_wealth'])
            wealths.append(wealth)
            roi = (wealth/1e6-1)
            RCs.append( roi* 100.0)
            RAs.append((np.power(roi+1, 1./9) -1)*100)
          
            dROIs.append((float(summary['wealth_ROI_mean']))*100)
            skews.append(float(summary['wealth_ROI_skew']))
            kurts.append(float(summary['wealth_ROI_kurt'][0]))
            
            sharpes.append(float(summary['wealth_ROI_Sharpe'])*100)
            sortinofs.append(float(summary['wealth_ROI_SortinoFull'])*100)
            
            JBs.append(float(summary['wealth_ROI_JBTest']))
            ADFs.append(float(summary['wealth_ROI_ADFTest']))
            
            CVaRFailRate = float(summary['CVaR_failRate']*100)
            VaRFailRate = float(summary['VaR_failRate']*100)
            CVaRFailRates.append(CVaRFailRate)
            VaRFailRates.append(VaRFailRate)
            
    
        wealths = np.asarray(wealths)
        RCs = np.asarray(RCs)
        RAs = np.asarray(RAs)
        dROIs =  np.asarray(dROIs)
        skews = np.asarray(skews)
        kurts = np.asarray(kurts)
        
        sharpes = np.asarray(sharpes)
        sortinofs = np.asarray(sortinofs) 
      
        JBs = np.asarray(JBs)
        ADFs = np.asarray(ADFs)
       
        CVaRFailRates = np.asarray(CVaRFailRates)
        VaRFailRates = np.asarray(VaRFailRates)
   
        #'$h$ & \alpha$ & $R_{C}$(\%) & $R_{A}$(\%) & $\mu$(\%) & $\sigma$(\%) & skew & kurt & $S_p$(\%) & $S_o$(\%)  & JB & ADF & $F_{CVaR}$(\%) & $F_{VaR}$(\%) \\ \hline \n')
        statIO.write("%s-%d-%4.2f & %4.2f & %4.2f & "%(
                    n_rv,
                    currentBestParam['period'],
                    float(currentBestParam['alpha']),
                    RCs.mean(), RAs.mean())
                    )
        
        statIO.write("%4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & "%(
                    dROIs.mean(), dROIs.std(),
                    skews.mean(), kurts.mean(),
                    sharpes.mean(), sortinofs.mean())
                    )
        
        statIO.write("%4.2e & %4.2e \\\ \hline\n"%(
                    max(JBs), max(ADFs))
#                     CVaRFailRates.mean(),
#                     VaRFailRates.mean())
                    )
        
        with open(outFile, 'ab') as fout:
            fout.write(statIO.getvalue())
        statIO.close()
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
    '''個股的統計分析
    '''
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
    statIO.write('rank & symbol & $R_{C}$(\%) & $R_{A}$(\%) & $\mu$(\%) & $\sigma$(\%) & skew & kurt & $S_p$(\%) & $S_o$(\%)  & JB & ADF \\\ \hline \n')
    
    for idx, symbol in enumerate(symbols):
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        tmp = df[startDate: endDate]
        rois = tmp['adjROI'].values

        mean = rois.mean()
        std = rois.std()
        skew = spstats.skew(rois)
        kurt = spstats.kurtosis(rois)
        sharpe = Performance.Sharpe(rois)
        sortinof, dd = Performance.SortinoFull(rois)
#         sortinop = Performance.SortinoPartial(rois)

        ret = sss.jarque_bera(rois)
        JB = ret[1]
        
        ret2 = sts.adfuller(rois)
        ADF = ret2[1]
        
        rtmp = rois/100 + 1
        rtmp[1] -= 0.001425 #buy fee
        rtmp[-1] -= 0.004425 #sell fee
        R_cum = rtmp[1:].prod() - 1 
        AR_cum = np.power((R_cum+1), 1./9) -1  
        
        #'rank & symbol & $R_{C}$ & $R_{A}$ $\mu$ & $\sigma$ & skew & kurt & JB & ADF & $S_p$ & $S_o$ 
        statIO.write('%2d & %s & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2e & %4.2e \\\ \hline \n'%(
                        idx+1, symbol, R_cum*100, AR_cum*100,  mean, std, skew, kurt,sharpe*100, sortinof*100,  JB, ADF ))
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
    
    statIO.write('symbol & $R_{C}$(\%) & $R_{A}$(\%) & $\mu$(\%) & $\sigma$(\%) & skew & kurt & $S_p$(\%) & $S_o$(\%)  & JB & ADF \\\ \hline \n')

    for idx, symbol in enumerate(symbols):
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        print symbol, df.columns
        tmp = df[startDate: endDate]
        rois = tmp['adjROI'].values

        mean = rois.mean()
        std = rois.std()
        skew = spstats.skew(rois)
        kurt = spstats.kurtosis(rois)
        sharpe = Performance.Sharpe(rois)
        sortinof, dd = Performance.SortinoFull(rois)
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
        
        statIO.write(' %s & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2f & %4.2e & %4.2e \\\ \hline \n'%(
                       symbol, R_cum*100, AR_cum*100,  mean, std, skew, kurt,sharpe*100, sortinof*100,  JB, ADF ))
        print symbol, R_cum, AR_cum
    
    resFile =  os.path.join(ExpResultsDir, 'comparison_daily_stats.txt')
    with open(resFile, 'wb') as fout:
        fout.write(statIO.getvalue())
        statIO.close()
    
    statIO.close()

def csv2Pkl(modelType="fixed"):
    n_rvs = range(5, 55, 5)
    hist_periods = range(50, 80, 10)
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95", "0.99")
    global ExpResultsDir
    if modelType == "fixed":
        myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", "LargestMarketValue_200501")
    elif modelType == "dynamic":
        myDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", "LargestMarketValue_200501_rv50")
    
    count = 0
    for n_rv in n_rvs:
        for period in hist_periods:
            for alpha in alphas:
                if modelType == "fixed":
                    dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                elif modelType == "dynamic":
                    dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                    
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                
                for exp in exps:
                    
                    for fileName in ['wealthProcess.csv', 'riskProcess.csv', 'actionProcess.csv']:
                        csvFile =os.path.join(exp, fileName)
                        procName = csvFile[csvFile.rfind('/')+1:csvFile.rfind('.')]
                        dfFile = os.path.join(exp, "%s.pkl"%(procName))
                        
                        if not os.path.exists(dfFile) and not os.path.exists(csvFile):
                            count += 1
                            print "rmdir[%s]:%s"%(count, exp)
#                             shutil.rmtree(exp)
                            break
                        
                        if os.path.exists(dfFile) and not os.path.exists(csvFile):
                            continue
                    
                        if not os.path.exists(dfFile) and  os.path.exists(csvFile):
                            df = pd.read_csv(csvFile, index_col=0, parse_dates=True)
                            df.save(dfFile)
     
                        #if transform successful
                        if os.path.exists(csvFile) and os.path.exists(dfFile):
                            os.remove(csvFile) 
                    
                    print exp


def y2yFixedSymbolResults():
    n_rvs = range(5, 55, 5)
    hist_periods = range(70, 130, 10)
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95")
#     n_rvs = [5,]
#     hist_periods = [70,]
#     alphas = ("0.5",)

    global ExpResultsDir
    
    myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", "LargestMarketValue_200501")
    for n_rv in n_rvs:
        t = time()
        avgIO = StringIO()        
        avgIO.write('run,startDate, endDate, n_stock, hist_period, alpha,wealth1, wealth1_std, wealth2, wealth2_std, wROI(%), wROI-std, JB, ADF,' )
        avgIO.write('meanROI(%%), meanROI-std, Sharpe(%%), Sharpe-std, SortinoFull(%%), SortinoF-std,SortinoPartial(%%),SortinoP-std,')
        avgIO.write(' downDevFull, downDevPartial, CVaRfailRate, VaRfailRate\n')
        
        for period in hist_periods:
            for alpha in alphas:
                dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                
                years = range(2005, 2013+1)
                wealth1, wealth2, rois = np.zeros((len(exps), len(years))), np.zeros((len(exps), len(years))),  np.zeros((len(exps), len(years)))
                sharpes, sortinofs, sortinops, dROIs = np.zeros((len(exps), len(years))),np.zeros((len(exps), len(years))),np.zeros((len(exps), len(years))),np.zeros((len(exps), len(years)))
                CVaRFailRates, VaRFailRates = np.zeros((len(exps), len(years))), np.zeros((len(exps), len(years)))
                downDevF, downDevP =  np.zeros((len(exps), len(years))), np.zeros((len(exps), len(years)))
                JBs, ADFs = np.zeros((len(exps), len(years))), np.zeros((len(exps), len(years)))
                
                for rdx, exp in enumerate(exps):
                    wealth_df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
                    risk_df = pd.read_pickle(os.path.join(exp, 'riskProcess.pkl'))
                    
                    for ydx, year in enumerate(years):     
                        startDate = date(year,1,3)
                        endDate = date(year, 12, 31)
                        
                        exp_df =  wealth_df[startDate:endDate]
                        exp_risk_df = risk_df[startDate:endDate]
                        
                        #wealth
                        wealth = exp_df.sum(axis=1)
                        wealth[-1] *=  (1-0.004425)
                                        
                        roi = (wealth[-1]/wealth[0] - 1)
                        wrois =  wealth.pct_change()
                        wrois[0] = 0
                    
                        wealth1[rdx,ydx] = wealth[0]
                        wealth2[rdx,ydx] = wealth[-1]
                        rois[rdx, ydx] = roi * 100
                        
                        #risk
                        dROI = wrois.mean()
                        sharpeVal = Performance.Sharpe(wrois)
                        sortinofVal, ddf = Performance.SortinoFull(wrois)
                        sortinopVal, ddp = Performance.SortinoPartial(wrois)
                        ret = sss.jarque_bera(wrois)
                        JB = ret[1]
                        ret2 = sts.adfuller(wrois)
                        ADF = ret2[1]
                        CVaRFailRate, VaRFailRate = VaRBackTest(exp_df,  exp_risk_df)
                        
                        dROIs[rdx, ydx] = dROI * 100
                        sharpes[rdx, ydx] = sharpeVal * 100
                        sortinofs[rdx, ydx] = sortinofVal * 100
                        sortinops[rdx, ydx] = sortinopVal*100
                        JBs[rdx, ydx] = JB 
                        ADFs[rdx, ydx] = ADF
                        downDevF[rdx, ydx] = ddf*100
                        downDevP[rdx, ydx] = ddp*100
                        CVaRFailRates[rdx, ydx] = CVaRFailRate*100
                        VaRFailRates[rdx, ydx] = VaRFailRate*100


#                 avgIO.write('run,startDate, endDate, n_stock, hist_period, alpha, wealth, wealth_std, wROI(%), wROI-std, JB, ADF,' )
#                 avgIO.write('meanROI(%%), meanROI-std, Sharpe(%%), Sharpe-std, SortinoFull(%%), SortinoF-std,SortinoPartial(%%),SortinoP-std,')
#                 avgIO.write(' downDevFull, downDevPartial, CVaRfailRate, VaRfailRate\n')

                for ydx, year in enumerate(years):
                    startDate = date(year,1,3)
                    endDate = date(year, 12, 31)
                        
                    exp_df =  wealth_df[startDate:endDate]
                    
                    avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,"%(len(exps),exp_df.index[0].strftime("%Y-%m-%d"), exp_df.index[-1].strftime("%Y-%m-%d"), n_rv, period, 
                            alpha, wealth1[:,ydx].mean(), wealth1[:,ydx].std(), wealth2[:,ydx].mean(), wealth2[:,ydx].std() ,rois[:, ydx].mean(), rois[:, ydx].std(), max(JBs[:, ydx]), max(ADFs[:, ydx])))
                    avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,"%(dROIs[:, ydx].mean(), dROIs[:, ydx].std(), sharpes[:,ydx].mean(), sharpes[:,ydx].std(),
                                                            sortinofs[:,ydx].mean(), sortinofs[:,ydx].std(), sortinops[:,ydx].mean(), sortinops[:,ydx].std()))
                    avgIO.write("%s,%s,%s,%s\n"%(max( downDevF[:,ydx]), max(downDevP[:,ydx]), max(CVaRFailRates[:,ydx]), max(VaRFailRates[:,ydx])))
                    print "n_rv:%s p:%s a:%s endDate:%s run:%s"%(n_rv, period, alpha, endDate, rdx+1)
                
        resFile =  os.path.join(ExpResultsDir, 'avg_y2yfixedSymbolSPPortfolio_n%s_result_2005.csv'%(n_rv))
        with open(resFile, 'ab') as fout:
            fout.write(avgIO.getvalue())
        avgIO.close()
        print "n_rv:%s OK, elapsed %.3f secs"%(n_rv, time()-t)


def y2yDynamicSymbolResults():
    n_rvs = range(20, 55, 5)
    hist_periods = range(50, 130, 10)
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95")
#     n_rvs = [5,]
#     hist_periods = [70,]
#     alphas = ("0.5",)

    global ExpResultsDir
    
    myDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", "LargestMarketValue_200501_rv50")
    for n_rv in n_rvs:
        t = time()
        avgIO = StringIO()        
        avgIO.write('run,startDate, endDate, n_stock, hist_period, alpha,wealth1, wealth1_std, wealth2, wealth2_std, wROI(%), wROI-std, JB, ADF,' )
        avgIO.write('meanROI(%%), meanROI-std, Sharpe(%%), Sharpe-std, SortinoFull(%%), SortinoF-std,SortinoPartial(%%),SortinoP-std,')
        avgIO.write(' downDevFull, downDevPartial, CVaRfailRate, VaRfailRate\n')
        
        for period in hist_periods:
            for alpha in alphas:
                dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                
                if len(exps) == 0:
                    continue
                
                years = range(2005, 2013+1)
                wealth1, wealth2, rois = np.zeros((len(exps), len(years))), np.zeros((len(exps), len(years))),  np.zeros((len(exps), len(years)))
                sharpes, sortinofs, sortinops, dROIs = np.zeros((len(exps), len(years))),np.zeros((len(exps), len(years))),np.zeros((len(exps), len(years))),np.zeros((len(exps), len(years)))
                CVaRFailRates, VaRFailRates = np.zeros((len(exps), len(years))), np.zeros((len(exps), len(years)))
                downDevF, downDevP =  np.zeros((len(exps), len(years))), np.zeros((len(exps), len(years)))
                JBs, ADFs = np.zeros((len(exps), len(years))), np.zeros((len(exps), len(years)))
                
                for rdx, exp in enumerate(exps):
                    wealth_df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
                    risk_df = pd.read_pickle(os.path.join(exp, 'riskProcess.pkl'))
                    
                    for ydx, year in enumerate(years):     
                        startDate = date(year,1,3)
                        endDate = date(year, 12, 31)
                        
                        exp_df =  wealth_df[startDate:endDate]
                        exp_risk_df = risk_df[startDate:endDate]
                        
                        #wealth
                        wealth = exp_df.sum(axis=1)
                        wealth[-1] *=  (1-0.004425)
                                        
                        roi = (wealth[-1]/wealth[0] - 1)
                        wrois =  wealth.pct_change()
                        wrois[0] = 0
                    
                        wealth1[rdx,ydx] = wealth[0]
                        wealth2[rdx,ydx] = wealth[-1]
                        rois[rdx, ydx] = roi * 100
                        
                        #risk
                        dROI = wrois.mean()
                        sharpeVal = Performance.Sharpe(wrois)
                        sortinofVal, ddf = Performance.SortinoFull(wrois)
                        sortinopVal, ddp = Performance.SortinoPartial(wrois)
                        ret = sss.jarque_bera(wrois)
                        JB = ret[1]
                        ret2 = sts.adfuller(wrois)
                        ADF = ret2[1]
                        CVaRFailRate, VaRFailRate = VaRBackTest(exp_df,  exp_risk_df)
                        
                        dROIs[rdx, ydx] = dROI * 100
                        sharpes[rdx, ydx] = sharpeVal * 100
                        sortinofs[rdx, ydx] = sortinofVal * 100
                        sortinops[rdx, ydx] = sortinopVal*100
                        JBs[rdx, ydx] = JB 
                        ADFs[rdx, ydx] = ADF
                        downDevF[rdx, ydx] = ddf*100
                        downDevP[rdx, ydx] = ddp*100
                        CVaRFailRates[rdx, ydx] = CVaRFailRate*100
                        VaRFailRates[rdx, ydx] = VaRFailRate*100


#                 avgIO.write('run,startDate, endDate, n_stock, hist_period, alpha, wealth, wealth_std, wROI(%), wROI-std, JB, ADF,' )
#                 avgIO.write('meanROI(%%), meanROI-std, Sharpe(%%), Sharpe-std, SortinoFull(%%), SortinoF-std,SortinoPartial(%%),SortinoP-std,')
#                 avgIO.write(' downDevFull, downDevPartial, CVaRfailRate, VaRfailRate\n')

                for ydx, year in enumerate(years):
                    startDate = date(year,1,3)
                    endDate = date(year, 12, 31)
                        
                    exp_df =  wealth_df[startDate:endDate]
                  
                    avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,"%(len(exps),exp_df.index[0].strftime("%Y-%m-%d"), exp_df.index[-1].strftime("%Y-%m-%d"), n_rv, period, 
                            alpha, wealth1[:,ydx].mean(), wealth1[:,ydx].std(), wealth2[:,ydx].mean(), wealth2[:,ydx].std() ,rois[:, ydx].mean(), rois[:, ydx].std(), max(JBs[:, ydx]), max(ADFs[:, ydx])))
                    avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,"%(dROIs[:, ydx].mean(), dROIs[:, ydx].std(), sharpes[:,ydx].mean(), sharpes[:,ydx].std(),
                                                            sortinofs[:,ydx].mean(), sortinofs[:,ydx].std(), sortinops[:,ydx].mean(), sortinops[:,ydx].std()))
                    avgIO.write("%s,%s,%s,%s\n"%(max( downDevF[:,ydx]), max(downDevP[:,ydx]), max(CVaRFailRates[:,ydx]), max(VaRFailRates[:,ydx])))
                    print "n_rv:%s p:%s a:%s endDate:%s run:%s"%(n_rv, period, alpha, endDate, rdx+1)
                
        resFile =  os.path.join(ExpResultsDir, 'avg_y2yDynamicSymbolSPPortfolio_n%s_result_2005.csv'%(n_rv))
        with open(resFile, 'ab') as fout:
            fout.write(avgIO.getvalue())
        avgIO.close()
        print "n_rv:%s OK, elapsed %.3f secs"%(n_rv, time()-t)

if __name__ == '__main__':
#     readWealthCSV()
#     parseFixedSymbolResults()
    parseBestFixedSymbol2Latex()
#     parseDynamicSymbolResults()
#     parseWCVaRSymbolResults()
#     individualSymbolStats()
#     groupSymbolStats()
#     comparisonStats()
#     csv2Pkl()
#     y2yFixedSymbolResults()
#     y2yDynamicSymbolResults()