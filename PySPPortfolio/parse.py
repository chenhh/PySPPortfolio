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
    sio = StringIO()
    sio.write('n_rv, hist_period, alpha, finalwealth, scen err\n')
    fullResFile = os.path.join(ExpResultsDir, 'fullyear_result.csv')
    
    pat = re.compile(r'final wealth:([\d.]+)')
    errPat = re.compile(r'generate scenario error count:(\d+)')
    for n_rv in n_rvs:
        for hist_period in hist_periods:
            for alpha in alphas:
                paramDir = os.path.join(ExpResultsDir, 
                            "n%s_h%s_s%s_a%s"%(n_rv, hist_period, n_scenario, alpha))
                
                expDirs = glob.glob(os.path.join(paramDir, 
                                "fixedSymbolSPPortfolio_20050103-20131231_*"))
                for expDir in expDirs:
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
                    print n_rv, hist_period, alpha, wealth, err 
                    sio.write('%s,%s,%s,%s,%s\n'%(n_rv, hist_period, 
                            alpha, wealth, err))
      
    with open(fullResFile, 'w') as fout:
        fout.write(sio.getvalue())
    sio.close()             
                    
                    
    

if __name__ == '__main__':
#     parseCSV2DataFrame()
#     testCSV()
#     readPkl()
    parseResults()