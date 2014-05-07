# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
from fixedSymbolSPPortfolio import constructModelMtx
from datetime import date

def testConstructModelMtx():
    symbols=('2330',) 
    startDate=date(2005,1,1)
    endDate=date(2005,1,10) 
    hist_period=5
    debug=False

    results = constructModelMtx(symbols, startDate, endDate, 
                      hist_period=hist_period, debug=debug)
    print len(results['allRiskyRetMtx'][0]), results['allRiskyRetMtx']
    print len(results['transDates']), results['transDates']
    print len(results['fullTransDates']), results['fullTransDates']
    
if __name__ == '__main__':
    testConstructModelMtx()