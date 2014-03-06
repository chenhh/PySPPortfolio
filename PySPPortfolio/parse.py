# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

'''
import os
import re
import glob
import platform
from cStringIO import StringIO
import numpy as np
from datetime import date
import pandas as pd
import SPATest

if platform.uname()[0] == 'Linux':
    ExpResultsDir =  os.path.join('/', 'home', 'chenhh' , 'Dropbox', 
                                  'financial_experiment', 'PySPPortfolio')

    
elif platform.uname()[0] =='Windows':
    ExpResultsDir= os.path.join('C:\\', 'Dropbox', 'financial_experiment', 
                                'PySPPortfolio')    
    
def parseCSV2DataFrame():
    currDir = os.getcwd()
    srcDir = os.path.join(currDir, 'pkl', 'marketvalue_csv', '*.csv')
    tgtDir = os.path.join(currDir, 'pkl', 'BasicFeatures')
    
    startDate, endDate = date(2004,1,1), date(2013, 12, 31)
    srcFiles = glob.glob(srcDir)
    for src in srcFiles:
        symbol = src[src.rfind('/')+1:src.rfind('.')]
        print symbol
#         if symbol in( "3045", '2412'):
#             continue
        df = pd.read_csv(open(src), index_col=("transDate",), parse_dates=True,
                         dtype={'transDate': date,
                                'openPrice': np.float,
                                'highestPrice': np.float,
                                'lowestPrie': np.float,
                                'closePrice': np.float,
                                'value': np.float,
                                'volume': np.float,
                                'adjROI': np.float})
#         mdf =  df[startDate:endDate]
        tgtFile = os.path.join(tgtDir, '%s.pkl'%(symbol))
        mdf.to_pickle(tgtFile)
        print symbol
        print mdf.head(10)
        print 
        
def testCSV():
    import csv
    currDir = os.getcwd()
    srcDir = os.path.join(currDir, 'pkl', 'marketvalue_csv',)
    srcFiles = [os.path.join(srcDir, name) for name in ('3045.csv', '2412.csv')]
    
    for name in srcFiles:
        reader = csv.DictReader(open(name), ['date', 'o', 'h', 'l', 'c', 'val', 'vol', 'r'])
        reader.next()
        for idx, line in enumerate(reader):
            print name, idx, float(line['r'])
    
def readPkl():
    currDir = os.getcwd()
    files = os.path.join(currDir, 'pkl', 'BasicFeatures', '*.pkl')
    for src in glob.glob(files):
        df = pd.load(src)
        print df.index[:10]
        print df.index[0].strftime('%Y%m%d')

def parseResults():
    n_rvs = (5, 10)
    hist_periods = (10, 20, 30 ,40 ,50 , 60 ,70 ,80)
    n_scenario = 200
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95", "0.99")
    
    pat = re.compile(r'final wealth:([\d.]+)')
    errPat = re.compile(r'generate scenario error count:(\d+)')
    for n_rv in n_rvs:
        sio = StringIO()
        latex = StringIO()
        sio.write('run, _n_rv, hist_period, alpha, rundime, finalwealth, ROI(%), scen err\n')
        fullResFile = os.path.join(ExpResultsDir, 'n%s_full_result.csv'%(n_rv))
        
        for alpha in alphas:
            latex.write('%s & '%(alpha))
            for hdx, hist_period in enumerate(hist_periods):
                paramDir = os.path.join(ExpResultsDir, 
                            "n%s_h%s_s%s_a%s"%(n_rv, hist_period, 
                                               n_scenario, alpha))
                
                expDirs = glob.glob(os.path.join(paramDir, 
                                "fixedSymbolSPPortfolio_20050103-20131231_*"))
                for rdx, expDir in enumerate(expDirs):
                    runTime = expDir[expDir.rfind('_')+1:]
                    
                    resFile = os.path.join(expDir, 'summary.txt')
                    if not os.path.exists(resFile):
                        continue
                    fin = open(resFile)
                    data = fin.read()        
                    
                    #parse summary file
                    #final wealth: wealth
                    m = pat.search(data)
                    merr = errPat.search(data)
                    wealth, err = 0, 0
                    if m:
                        wealth = float(m.group(1))
                    if merr:
                        err = int(merr.group(1))
                    roi = (wealth/1e6 -1)*100
                    #只取第一輪的資料
                    if rdx == 0 and hdx != len(hist_periods)-1:
                        if err <= 30:
                            latex.write('%.2f & '%(roi))
                        else:
                            latex.write('- & ')
                    elif rdx == 0 and hdx == len(hist_periods)-1:
                        if err <= 30:
                            latex.write('%.2f \\\ \hline\n'%(roi))
                        else:
                            latex.write('- \\\ \hline \n ')
                        
#                     print n_rv, hist_period, alpha, runTime, wealth, roi, err 
                    sio.write('%s,%s,%s,%s,%s,%s,%s,%s\n'%(rdx+1, n_rv, hist_period, 
                            alpha,runTime, wealth, roi, err))
      
        print latex.getvalue()
        with open(fullResFile, 'w') as fout:
            fout.write(sio.getvalue())
        sio.close()             


def benchmark():
    currDir = os.getcwd()
    symbols = ['2330', '2317', '6505', '2412', '2454',
#                 '2882', '1303', '1301', '1326', '2881'
               ]
    startDate, endDate = date(2005,1, 1), date(2013, 12, 31)
    
    cumROIs = []
    for src in symbols:
        wealth = 1e6
        fin = os.path.join(currDir, 'pkl', 'BasicFeatures', '%s.pkl'%(src))
        df = pd.read_pickle(fin)
        data = df[startDate: endDate]
#         print src, data.index[0], data.index[-1]
        rois = data['adjROI']
#         print data.index[0], data.index[-1]
        for roi in rois[:-1]:
            roi = roi/100.0
            wealth *= (1+roi)
        cumROI = (wealth/1e6-1) * 100
        ar = ((np.power((1+cumROI/100), 1./9) -1 ))*100
        cumROIs.append(cumROI)
        print "buy \& hold, %s, %s, %.2f, %.2f \\\ \hline"%(src, src, cumROI, ar)
#     mu = np.mean(cumROIs)
    mu = 188.84
    ar_mu = ((np.power((1+mu/100), 1./9) -1 ))*100
    print "avg: %.2f, %.2f \\\ \hline"%( mu, ar_mu)

def runSPATest():
    currDir = os.getcwd()
    symbols = ['2330', '2317', '6505', '2412', '2454',
                '2882', '1303', '1301', '1326', '2881'
               ]
    startDate, endDate = date(2005,1, 1), date(2013, 12, 31)
    n_rvs = (5, 10)
    hist_periods = (20, 30 ,40 ,50 , 60 ,70 ,80)
    n_scenario = 200
    alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", "0.75", "0.8", 
              "0.85", "0.9", "0.95")
    
    for n_rv in n_rvs:
        sio = StringIO()
        latex = StringIO()
        sio.write('run, n_rv, hist_period, alpha, finalwealth, ROI(%), scen err, SPAtest\n')
        fullResFile = os.path.join(ExpResultsDir, 'n%s_full_SPA_result.csv'%(n_rv))
        
        for alpha in alphas:
            latex.write('%s & '%(alpha))
            for hdx, hist_period in enumerate(hist_periods):
                paramDir = os.path.join(ExpResultsDir, 
                            "n%s_h%s_s%s_a%s"%(n_rv, hist_period, 
                                               n_scenario, alpha))
                
                expDirs = glob.glob(os.path.join(paramDir, 
                                "fixedSymbolSPPortfolio_20050103-20131231_*"))
                for rdx, expDir in enumerate(expDirs):
                    runTime = expDir[expDir.rfind('_')+1:]
                    
                    wealthPkl = os.path.join(expDir, 'wealthProcess.pkl')
                    depositPkl = os.path.join(expDir, 'depositProcess.pkl')
                    if not os.path.exists(wealthPkl) or not os.path.exists(depositPkl):
                        continue
                    wealth =  pd.read_pickle(wealthPkl)
                    deposit = pd.read_pickle(depositPkl)
                    #combine
                    wealth['deposit'] = deposit
                    tWealth = wealth.sum(axis=1)
#                     print tWealth
                    rois = [0,]
                    for idx, w in enumerate(tWealth[1:]):
                        roi = float(w)/tWealth[idx] - 1.
                        rois.append(roi)
                    #construct diff obj
                    rois = np.asarray(rois)
                    diffObj = SPATest.DiffObject2(rois)
                    pval = SPATest.RCTest(diffObj, n_samplings=5000)
     
if __name__ == '__main__':
#     parseCSV2DataFrame()
#     testCSV()
#     readPkl()
#     parseResults()
#     benchmark()
    runSPATest()