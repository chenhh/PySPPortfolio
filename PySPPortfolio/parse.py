# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

'''
import os
import glob
import numpy as np
from datetime import date
import pandas as pd


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
        
if __name__ == '__main__':
#     parseCSV2DataFrame()
#     testCSV()
    readPkl()