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
 


def parseSymbolResults(modelType = "fixed"):
    '''whole period'''
    
    if modelType == "fixed":
        n_rvs = range(5, 55, 5)
        hist_periods = range(50, 130, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95", '0.99')
        myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", "LargestMarketValue_200501")
        
    elif modelType == "dynamic":
        n_rvs = range(5, 55, 5)
        hist_periods = range(90, 120+10, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7")
        myDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", "LargestMarketValue_200501_rv50")
       
    
    for n_rv in n_rvs:
        t = time()
        avgIO = StringIO()        
        avgIO.write('run, n_rv, period, alpha, time, wealth, wealth-std, wROI(%), wROI-std,' )
        avgIO.write('dROI(%%), stdev, skew, kurt, Sp(%%), Sp-std, StF(%%), StF-std,')
        avgIO.write('StP(%%), Stp-std, downDevF, downDevP,  JB, ADF, CVaRfailRate, VaRfailRate, scen err\n')
        
        for period in hist_periods:
            if n_rv == 50 and period == 50:
                continue
            
            
            for alpha in alphas:
                if modelType == "fixed":
                    dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                elif modelType == "dynamic":
                    dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                    
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                wealths, ROI_Cs, dROIs, stdevs, skews, kurts =[], [], [], [], [], []
                JBs, ADFs = [], []
                sharpes, sortinofs, sortinops,  downDevF, downDevP = [],[],[],[],[]
                CVaRFailRates, VaRFailRates = [], []
                elapsed, scenerr = [], []
                
                if len(exps) > 3:
                    exps = exps[:3]
                  
                if len(exps) == 0:
                    avgIO.write('NA,'*26 + '\n')
                    continue
                    
                for edx, exp in enumerate(exps):
                    print exp
                    summaryFile = os.path.join(exp, "summary.json")
                    summary = json.load(open(summaryFile))         
                    print dirName
                    
                    #wealth and cum ROI
                    wealth = float(summary['final_wealth'])
                    wealths.append(wealth)
                    ROI_Cs.append((wealth/1e6-1) * 100.0)
                    
                    elapsed.append(float(summary['elapsed']))
                    scenerr.append(summary['scen_err_cnt'])
                    try:
                        dROIs.append(float(summary['wealth_ROI_mean'])*100)
                        stdevs.append(float(summary['wealth_ROI_stdev'])*100)
                        skews.append(float(summary['wealth_ROI_skew']))
                        kurts.append(float(summary['wealth_ROI_kurt']))
                        sharpes.append(float(summary['wealth_ROI_Sharpe'])*100)
                        sortinofs.append(float(summary['wealth_ROI_SortinoFull'])*100)
                        sortinops.append(float(summary['wealth_ROI_SortinoPartial'])*100)
                        downDevF.append((float(summary['wealth_ROI_downDevFull']))*100)
                        downDevP.append((float(summary['wealth_ROI_downDevPartial']))*100)
                        JBs.append(float(summary['wealth_ROI_JBTest']))
                        ADFs.append(float(summary['wealth_ROI_ADFTest']))
                        
                    except (KeyError,TypeError):
                        #read wealth process
                        print "read raw df n_rv-period-alpha: %s-%s-%s:%s"%(n_rv, period, alpha, edx+1)
                        df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
        
                        proc = df.sum(axis=1)
                        wrois =  proc.pct_change()
                        wrois[0] = 0
                        
                        dROI = wrois.mean()
                        dROIs.append(dROI*100)
                        summary['wealth_ROI_mean'] = dROI
                        
                        stdev = wrois.std()
                        stdevs.append(stdev)
                        summary['wealth_ROI_stdev'] = stdev
                        
                        skew = spstats.skew(wrois)
                        skews.append(skew)
                        summary['wealth_ROI_skew'] = skew
                        
                        kurt = spstats.kurtosis(wrois)
                        kurts.append(kurt) 
                        summary['wealth_ROI_kurt'] = kurt
                      
                        sharpe = Performance.Sharpe(wrois)
                        sharpes.append(sharpe*100)
                        summary['wealth_ROI_Sharpe'] = sharpe 
                        
                        sortinof, ddf = Performance.SortinoFull(wrois)
                        sortinofs.append(sortinof*100)
                        downDevF.append(ddf*100)
                        summary['wealth_ROI_SortinoFull'] = sortinof
                        summary['wealth_ROI_downDevFull'] = ddf
                        
                        sortinop, ddp = Performance.SortinoPartial(wrois)
                        sortinops.append(sortinop*100)
                        downDevP.append(ddp*100)
                        summary['wealth_ROI_SortinoPartial'] = sortinop
                        summary['wealth_ROI_downDevPartial'] = ddp
                        
                        ret = sss.jarque_bera(wrois)
                        JB = ret[1]
                        JBs.append(JB)
                        summary['wealth_ROI_JBTest'] = JB
        
                        ret2 = sts.adfuller(wrois)
                        ADF = ret2[1]
                        ADFs.append(ADF)
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
                        wealth_df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
                        risk_df = pd.read_pickle(os.path.join(exp, 'riskProcess.pkl'))
                        
                        CVaRFailRate, VaRFailRate = VaRBackTest(wealth_df, risk_df)
                        CVaRFailRates.append(CVaRFailRate*100)
                        VaRFailRates.append(VaRFailRate*100)
                        summary['VaR_failRate'] = VaRFailRate
                        summary['CVaR_failRate'] = CVaRFailRate
                       
                        print "CVaR fail:%s, VaR fail:%s"%(CVaRFailRate, VaRFailRate)
                       
                        fileName = os.path.join(exp, 'summary.json')
                        with open (fileName, 'w') as fout:
                            json.dump(summary, fout, indent=4)
                     
        
                wealths = np.asarray(wealths)    
                ROI_Cs = np.asarray(ROI_Cs)
                dROIs =  np.asarray(dROIs)
                stdevs = np.asarray(stdevs)
                skews = np.asarray(skews)
                kurts = np.asarray(kurts)
                JBs = np.asarray(JBs)
                ADFs = np.asarray(ADFs)
                
                sharpes = np.asarray(sharpes)
                sortinofs = np.asarray(sortinofs) 
                sortinops = np.asarray(sortinops)
                downDevF = np.asarray(downDevF)
                downDevP = np.asarray(downDevP)
                
                CVaRFailRates = np.asarray(CVaRFailRates)
                VaRFailRates = np.asarray(VaRFailRates)
               
                elapsed = np.asarray(elapsed)
                scenerr = np.asarray(scenerr)
                           
                avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,"%(len(ROI_Cs), n_rv, period, alpha,  elapsed.mean(),
                                wealths.mean(), wealths.std(),  ROI_Cs.mean(), ROI_Cs.std() ))
                avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,"%(dROIs.mean(), stdevs.mean(), skews.mean(),kurts.mean(), 
                                sharpes.mean(), sharpes.std(), sortinofs.mean(), sortinofs.std() )) 
                avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%( sortinops.mean(), sortinops.std(), downDevF.mean(), 
                                                             downDevP.mean(), max(JBs), max(ADFs), CVaRFailRates.mean(), 
                                                             VaRFailRates.mean(),scenerr.mean() ))
                    
       
        if modelType == "fixed":
            resFile =  os.path.join(ExpResultsDir, 'avg_fixedSymbolSPPortfolio_n%s_result_2005.csv'%(n_rv))
        elif modelType == "dynamic":
            resFile =  os.path.join(ExpResultsDir, 'avg_dynamicSymbolSPPortfolio_n%s_result_2005.csv'%(n_rv))
                
        with open(resFile, 'wb') as fout:
            fout.write(avgIO.getvalue())
        avgIO.close()
        print "n_rv:%s OK, elapsed %.3f secs"%(n_rv, time()-t)


def parseBestSymbol2Latex(modelType = "fixed"):
    
    if modelType == "fixed":
        n_rvs = range(5, 55, 5)
        hist_periods = range(70, 130, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95")
        myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", 
                             "LargestMarketValue_200501")
        outFile = os.path.join(ExpResultsDir, "fixedSymbolBestParam.txt")
        
    elif modelType == "dynamic":
        n_rvs = range(5, 55, 5)
        hist_periods = range(90, 120+10, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7")
        myDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", 
                             "LargestMarketValue_200501_rv50")
        outFile = os.path.join(ExpResultsDir, "dynamicSymbolBestParam.txt")
    
    global ExpResultsDir
  
    
    for n_rv in n_rvs:
        t = time()
       
        statIO = StringIO()
        if not os.path.exists(outFile):        
            statIO.write('$n-h-\alpha$ & $R_{C}$(\%) & $R_{A}$(\%) & ')
            statIO.write('$\mu$(\%) & $\sigma$(\%) & skew & kurt & ')
            statIO.write('$S_p$(\%) & $S_t$(\%)  & JB & ADF  \\\ \hline \n')
        
        currentBestParam = {"period": 0, "alpha": 0, "wealths": 0}
        for period in hist_periods:
            for alpha in alphas:
                if modelType == "fixed":
                    dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                elif modelType == "dynamic":
                    dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                    
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
        print "n_rv:%s bestParam p:%s a:%s"%(n_rv, currentBestParam['period'], 
                                             currentBestParam['alpha'])
        if modelType == "fixed":
            dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, 
                        currentBestParam['period'], currentBestParam['alpha'])
        elif modelType == "dynamic":
            dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, 
                        currentBestParam['period'], currentBestParam['alpha'])
            
        exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
        wealths, RAs,RCs = [], [], []
        dROIs, stdevs, skews, kurts = [], [], [], []
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
            stdevs.append(float(summary['wealth_ROI_stdev'])*100)
            skews.append(float(summary['wealth_ROI_skew']))
            kurts.append(float(summary['wealth_ROI_kurt']))
            
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
        stdevs = np.asarray(stdevs)
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
        
        statIO.write("%4.4f & %4.4f & %4.2f & %4.2f & %4.2f & %4.2f & "%(
                    dROIs.mean(), stdevs.mean(),
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
     



def comparisonStats():
    symbols = [
        'TAIEX', '0050',
    ]
    
    startDate=date(2005,1,3)
    endDate=date(2013,12,31)
    
    statIO = StringIO()   
    
    statIO.write('symbol & $R_{C}$(\%) & $R_{A}$(\%) & ')
    statIO.write('$\mu$(\%) & $\sigma$(\%) & skew & kurt & ')
    statIO.write('$S_p$(\%) & $S_o$(\%)  & JB & ADF \\\ \hline \n')

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


def y2yResults(modelType="fixed"):
    '''
    '''
    
    global ExpResultsDir
    if modelType == "fixed":
        n_rvs = range(5, 55, 5)
        hist_periods = range(50, 130, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95")
        myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", "LargestMarketValue_200501")
        
    elif modelType == "dynamic":
        n_rvs = range(5, 55, 5)
        hist_periods = range(90, 120+10, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7")
        myDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", "LargestMarketValue_200501_rv50")
   
    for n_rv in n_rvs:
        t = time()    
        avgIO = StringIO()        
        avgIO.write('run, startDate, endDate, n_rv, period, alpha,  w1, w1-std, w2, w2-std, wROI(%), wROI-std,' )
        avgIO.write('dROI(%%), stdev, skew, kurt, Sp(%%), Sp-std, StF(%%), StF-std,')
        avgIO.write('StP(%%), Stp-std, downDevF, downDevP,  JB, ADF, CVaRfailRate, VaRfailRate, scen err\n')
        
        for period in hist_periods:
            if n_rv == 50 and period == 50:
                continue
            
            for alpha in alphas:
                if modelType == "fixed":
                    dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                elif modelType == "dynamic":
                    dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                     
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                if len(exps) > 3:
                    exps = exps[:3]
                
                years = range(2005, 2013+1)
                d1, d2 = len(exps), len(years)
                
                wealth1, wealth2, ROI_Cs = np.zeros((d1, d2)), np.zeros((d1, d2)),  np.zeros((d1, d2))
                dROIs, stdevs, skews, kurts = np.zeros((d1, d2)),  np.zeros((d1, d2)), np.zeros((d1, d2)), np.zeros((d1, d2))
                JBs, ADFs =  np.zeros((d1, d2)),  np.zeros((d1, d2))
                sharpes =  np.zeros((d1, d2))
                sortinops, downDevP =  np.zeros((d1, d2)),  np.zeros((d1, d2))
                sortinofs,downDevF =  np.zeros((d1, d2)),  np.zeros((d1, d2))
                CVaRFailRates, VaRFailRates =   np.zeros((d1, d2)),  np.zeros((d1, d2))
               
                for edx, exp in enumerate(exps):
                    wealth_df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
                    risk_df = pd.read_pickle(os.path.join(exp, 'riskProcess.pkl'))
                    
                    for ydx, year in enumerate(years):     
                        startDate = date(year,1,1)
                        endDate = date(year, 12, 31)
                        
                        exp_wealth_df =  wealth_df[startDate:endDate]
                        exp_risk_df = risk_df[startDate:endDate]
                        
                        #wealth
                        wealth = exp_wealth_df.sum(axis=1)
                        wealth[-1] *=  (1-0.004425)
                        wealth1[edx,ydx] = wealth[0]
                        wealth2[edx,ydx] = wealth[-1]
                        
                        #cum ROI
                        roi = (wealth[-1]/wealth[0] - 1)
                        wrois =  wealth.pct_change()
                        wrois[0] = 0
                        ROI_Cs[edx, ydx] = roi * 100
                        
                        #stats
                        dROIs[edx, ydx] = wrois.mean() * 100
                        stdevs[edx, ydx] = wrois.std()*100
                        skews[edx, ydx] = spstats.skew(wrois)
                        kurts[edx, ydx] = spstats.kurtosis(wrois)
                        
                        #JB, ADF
                        ret = sss.jarque_bera(wrois)
                        JB = ret[1]
                        ret2 = sts.adfuller(wrois)
                        ADF = ret2[1]
                        JBs[edx, ydx] = JB 
                        ADFs[edx, ydx] = ADF
                        
                        #Sharpe
                        sharpe = Performance.Sharpe(wrois)
                        sharpes[edx, ydx] = sharpe * 100
                        
                        sortinof, ddf = Performance.SortinoFull(wrois)
                        sortinofs[edx, ydx] = sortinof * 100
                        downDevF[edx, ydx] = ddf*100
                        
                        sortinop, ddp = Performance.SortinoPartial(wrois)
                        sortinops[edx, ydx] = sortinop*100
                        downDevP[edx, ydx] = ddp*100
                      
                        CVaRFailRate, VaRFailRate = VaRBackTest(exp_wealth_df,  exp_risk_df)
                        CVaRFailRates[edx, ydx] = CVaRFailRate*100
                        VaRFailRates[edx, ydx] = VaRFailRate*100
                      
                for ydx, year in enumerate(years):
                    startDate = date(year,1,1)
                    endDate = date(year, 12, 31)
                        
                    exp_df =  wealth_df[startDate:endDate]
                    
                    #avgIO.write('run, startDate, endDate, n_rv, period, alpha,  w1, w1-std, w2, w2-std, wROI(%), wROI-std,' )
                    #avgIO.write('dROI(%%), stdev, skew, kurt, Sp(%%), Sp-std, StF(%%), StF-std,')
                    #avgIO.write('StP(%%), Stp-std, downDevF, downDevP,  JB, ADF, CVaRfailRate, VaRfailRate\n')
                    
                    avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"%(
                            len(exps),exp_df.index[0].strftime("%Y-%m-%d"), 
                            exp_df.index[-1].strftime("%Y-%m-%d"), 
                            n_rv, period, alpha, 
                            wealth1[:,ydx].mean(), wealth1[:,ydx].std(), 
                            wealth2[:,ydx].mean(), wealth2[:,ydx].std(), 
                            ROI_Cs[:, ydx].mean(), ROI_Cs[:, ydx].std(), 
                            ))
                    
                    avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,"%(
                            dROIs[:, ydx].mean(),  
                            stdevs[:, ydx].mean(),
                            skews[:, ydx].mean(), 
                            kurts[:, ydx].mean(), 
                            sharpes[:,ydx].mean(), sharpes[:,ydx].std(),
                            sortinofs[:,ydx].mean(), sortinofs[:,ydx].std() 
                           ))
                    avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s\n"%(
                            sortinops[:,ydx].mean(), sortinops[:,ydx].std(),
                            downDevF[:,ydx].mean(), downDevP[:,ydx].mean(),
                            max(JBs[:,ydx]), max(ADFs[:,ydx]),
                            CVaRFailRates[:,ydx].mean(), VaRFailRates[:,ydx].mean()))
             
                    print "n_rv:%s p:%s a:%s endDate:%s run:%s"%(n_rv, period, alpha, endDate, edx+1)
        
        if modelType == "fixed":
            resFile =  os.path.join(ExpResultsDir, 'avg_y2yfixedSymbolSPPortfolio_n%s_result_2005.csv'%(n_rv))
        elif modelType == "dynamic":
            resFile =  os.path.join(ExpResultsDir, 'avg_y2ydynamicSymbolSPPortfolio_n%s_result_2005.csv'%(n_rv))
                
        with open(resFile, 'ab') as fout:
            fout.write(avgIO.getvalue())
        avgIO.close()
        print "n_rv:%s OK, elapsed %.3f secs"%(n_rv, time()-t)


def compareY2YResults():
    '''
    comparing the SP, SIP, and buy-and-hold 
    fixedSymbol best parameters
    5-100-0.60, 10-80-0.50 , 15-80-0.50, 20-110-0.50, 
    25-100-0.55,30-120-0.60, 35-120-0.50, 40-110-0.50, 
    45-120-0.55,, 50-120-0.50
    
    dynamic best parameters:
    5-120-0.50 10-120-0.55  15-120-0.50  20-120-0.60 
    25-120-0.55  30-120-0.50  35-110-0.50  40-120-0.50  
    45-110-0.50  50-110-0.55 
    '''
    n_rvs = range(5, 50+5, 5)
    fixedParams = [ 
            (5, 100,0.60), (10, 80, 0.50) , (15, 80, 0.50), (20, 110, 0.50), 
            (25, 100,0.55), (30, 120, 0.60), (35,120,0.50), (40,110,0.50), 
            (45,120, 0.55), (50,120,0.50)
    ] 
    dynamParams = [
            (5,120,0.50), (10,120,0.55),  (15,120,0.50),  (20,120,0.60), 
            (25,120,0.55),  (30,120,0.50),  (35,110,0.50),  (40,120,0.50),  
            (45,110,0.50),  (50,110,0.55) 
    ]
    
    comps = zip(n_rvs, fixedParams, dynamParams)
    
    BHDir = os.path.join(ExpResultsDir, "BuyandHoldPortfolio")
    fixedDir  = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", 
                             "LargestMarketValue_200501")
    dynDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", 
                              "LargestMarketValue_200501_rv50")
       
    avgIO = StringIO()        
    avgIO.write('startDate, endDate, n_rv, period, alpha,  w1, w1-std, w2, w2-std, wROI(%), wROI-std,' )
    avgIO.write('dROI(%%), stdev, skew, kurt, Sp(%%), Sp-std, StF(%%), StF-std,')
    avgIO.write('StP(%%), Stp-std, downDevF, downDevP,  JB, ADF, CVaRfailRate, VaRfailRate, scen err\n')
           
    for n_rv, fixed, dyn in comps:
        bh_wealths =  pd.read_pickle(os.path.join(BHDir,"wealthSum_n%s.pkl"%(n_rv)))
        bh_rois = bh_wealths.pct_change()
        bh_rois[0] = 0
        
        #fixed
        fixed_expDir = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(fixed[0],fixed[1],fixed[2])
        fixedExps = glob(os.path.join(fixedDir, fixed_expDir, "20050103-20131231_*"))
        fixed_wealths, fixed_rois = [], []
        for edx, exp in enumerate(fixedExps[:3]):
            df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
            wealths = df.sum(axis=1)
            fixed_wealths.append(wealths)
           
            rois = wealths.pct_change()
            rois[0]= 0
            fixed_rois.append(rois)
            
         
        #dynamic 
        dyn_expDir = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(dyn[0],dyn[1],dyn[2])
        dynExps = glob(os.path.join(dynDir, dyn_expDir, "20050103-20131231_*"))
        dyn_wealths, dyn_rois = [], []
        for edx, exp in enumerate(dynExps[:3]):
            df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
            wealths = df.sum(axis=1)
            dyn_wealths.append(wealths)
            
            rois = wealths.pct_change()
            rois[0]= 0
            dyn_rois.append(rois)
        
        #stats
      
        years = range(2005, 2013+1)
        d1, d2 = len(exps), len(years)
                

def _ROIstats(rois):
    mu = rois.mean()
    stdev = rois.std()
    skew = spstats.skew(rois) 
    kurt = spstats.kurtosis(rois)
  
    sharpe = Performance.Sharpe(rois)
    sortinof, ddf = Performance.SortinoFull(rois)
    sortinop, ddp = Performance.SortinoPartial(rois)
   
    ret = sss.jarque_bera(rois)
    JB = ret[1]
    
    ret2 = sts.adfuller(rois)
    ADF = ret2[1]
    return {"mu":mu, "stdev":stdev,
            "skew":skew, "kurt":kurt,
            "sharpe":sharpe, "sortinof":sortinof,
            "sortinop":sortinop, "ddf":ddf,
            "ddp":ddp}
          

if __name__ == '__main__':
#     readWealthCSV()
#     parseSymbolResults(modelType = "fixed")
#     parseSymbolResults(modelType = "dynamic")    
#     parseBestSymbol2Latex(modelType = "fixed")
#     parseBestSymbol2Latex(modelType = "dynamic")

#     parseWCVaRSymbolResults()
#     individualSymbolStats()
#     groupSymbolStats()
#     comparisonStats()
#     csv2Pkl()
    y2yResults("fixed")
    y2yResults("dynamic")
#     compareY2YResults()
