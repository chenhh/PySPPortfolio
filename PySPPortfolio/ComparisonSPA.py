# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
Comparison results by SPA
'''
from __future__ import division
from stats import SPATest
import numpy as np
import time
import os
from glob import glob
import sys
import pandas as pd
ProjectDir = os.path.join(os.path.abspath(os.path.curdir), '..')
sys.path.insert(0, ProjectDir)
from PySPPortfolio import  (PklBasicFeaturesDir,  ExpResultsDir)
import simplejson as json
from datetime import date
from cStringIO import StringIO
 
class ROIDiffObject(object):
    
    def __init__(self, bhROIs):
        '''
        @bhROIs, np.array, rois of buy and hold strategy
        '''
       
        self.bhROIs = np.asmatrix(bhROIs)        
        self.n_periods = len(bhROIs) 
        self.n_rules = 0
        
        #n_rules * n_periods
        self.ruleSignalMatrix = None    
    
    def setCompareROIs(self, ROIs):
        '''
        @ROIs: np.array
        '''
        assert len(ROIs) == self.n_periods
      
        if self.ruleSignalMatrix is None:  
            self.ruleSignalMatrix = np.asmatrix(ROIs)
        else: 
            self.ruleSignalMatrix = np.vstack((self.ruleSignalMatrix, 
                                                np.asmatrix(ROIs)))
        self.n_rules += 1
        
    def _kernelWeight(self, P, Q):
        '''
        K(T,P) = (T-P)/T*(1-Q)**P + P/T*(1-Q)**(T-P)
        @param P (pos): P-th period 
        note in the paper, the index of first element is 1, not zero.
        '''
        T = self.n_periods
        return (T - P)/T*(1.-Q)**P + P/T*(1.-Q)**(T-P)
    

    def getROIDiffMatrix(self):
        '''
          
        @return diffMatrix, shape: n_rules * n_periods
        avg(L[k]) > 0, the base model is inferior to kth model
        avg(L[k]) < 0, the base model is superior to kth model
        
        the above definition is consistent with the null hypothesis
        ''' 
        diffMatrix =self.ruleSignalMatrix -  self.bhROIs
        return  diffMatrix


    def getDeviation(self, diffMtx, Q):
        ''' 
        sampling deviation of each rule
        @diffMtrix: numpy.matrix, row: different value of ith rules, column index: period
        @return column matrix of each rule deviation, shape: n_rules*1 
        '''
        weights = [0.,]
        weights.extend(self._kernelWeight(pos, Q) for pos in range(1, self.n_periods))
        
        varColMtx = np.var(diffMtx, axis=1) 
        D = diffMtx - diffMtx.mean(axis=1)
        
        #auto-covariance(from 1 to length-1)
        for pt in range(1, self.n_periods):
            varColMtx = varColMtx + weights[pt] * np.asmatrix(
                np.diagonal( D[:, :self.n_periods-pt] * D.T[pt:, :] / self.n_periods )).T
       
        #shape: n_rules *1
        return np.sqrt(varColMtx)


def SPA4BHSymbol(modelType="fixed", years=None):
    '''
    buy-and-hold versus. SP model
    '''
    t0 = time.time()
    
    if modelType == "fixed":
        n_rvs = range(5, 55, 5)
        hist_periods = range(50, 130, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95")
        myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", 
                             "LargestMarketValue_200501")
        if not years:
            resFile = os.path.join(ExpResultsDir, "SPA", 
                                   "SPA_Fixed_BetterBH.csv")
        else:
            resFile = os.path.join(ExpResultsDir, "SPA", 
                                   "SPA_year_Fixed_BetterBH.csv")
        
        
    elif modelType == "dynamic":
        n_rvs = range(5, 55, 5)
        hist_periods = range(90, 120+10, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7")
        myDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", 
                             "LargestMarketValue_200501_rv50")
        if not years:
            resFile = os.path.join(ExpResultsDir, "SPA", 
                               "SPA_Dynamic_BetterBH.csv")
        else:
            resFile = os.path.join(ExpResultsDir, "SPA", 
                               "SPA_year_Dynamic_BetterBH.csv")

    bhDir = os.path.join(ExpResultsDir, "BuyandHoldPortfolio")
    
    
    #stats file
    avgIO = StringIO()
    if not os.path.exists(resFile):
        if not years: 
            avgIO.write('n_rv, SPA_Q, sampling, n_rule, n_period, P-value\n')
        else:
            avgIO.write('startDate, endDate, n_rv, SPA_Q, sampling, n_rule, n_period, P-value\n')
            
    bh_rois = None
    tgt_rois = []
    
    for n_rv in n_rvs:
        t1 = time.time()
        
        #load buyhold ROI
        bh_df = pd.read_pickle(os.path.join(bhDir,"wealthSum_n%s.pkl"%n_rv))
        bh_finalWealth = bh_df[-1]
        bh_rois = bh_df.pct_change()
        bh_rois[0] = 0
        
        #load model ROI
        for period in hist_periods:
            for alpha in alphas:
                if modelType == "fixed":
                    dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                elif modelType == "dynamic":
                    dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                    
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                if len(exps) > 3:
                    exps = exps[:3]
                    
                for exp in exps:
                    #load comparison rois
                    df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
                    proc = df.sum(axis=1)
                    exp_finalWealth = proc[-1]
                    
                    if exp_finalWealth >= bh_finalWealth:
                        wrois =  proc.pct_change()
                        wrois[0] = 0
                        tgt_rois.append(wrois)
                  
        print " SPA4BHSymbol n_rv: %s, load data OK, %.3f secs"%(n_rv, time.time()-t1)
        
        if not years: 
            #set diff obj
            diffobj = ROIDiffObject(bh_rois)
            for rois in tgt_rois:
                diffobj.setCompareROIs(rois)
       
            #SPA test
            t2 = time.time()
            Q = 0.5
            n_samplings = 5000
            verbose = True
            pvalue = SPATest.SPATest(diffobj, Q, n_samplings, "SPA_C", verbose)
            print "full n_rv:%s, (n_rules, n_periods):(%s, %s), SPA_C:%s elapsed:%.3f secs"%(
                    n_rv, diffobj.n_rules, diffobj.n_periods, pvalue, time.time()-t2)
            avgIO.write("%s,%s,%s,%s,%s,%s\n"%(n_rv, Q, n_samplings, 
                                diffobj.n_rules, diffobj.n_periods, pvalue))
        else:
            for year in years:
                startDate, endDate = date(year, 1, 1), date(year, 12, 31)
                
                diffobj = ROIDiffObject(bh_rois[startDate:endDate])
                for rois in tgt_rois:
                    diffobj.setCompareROIs(rois[startDate:endDate])
                    
                t2 = time.time()
                Q = 0.5
                n_samplings = 5000
                verbose = True
                pvalue = SPATest.SPATest(diffobj, Q, n_samplings, "SPA_C", verbose)
                print "year:%s n_rv:%s, (n_rules, n_periods):(%s, %s), SPA_C:%s elapsed:%.3f secs"%(
                        year, n_rv, diffobj.n_rules, diffobj.n_periods, pvalue, time.time()-t2)
                avgIO.write("%s, %s,%s,%s,%s,%s,%s,%s\n"%(
                    bh_rois[startDate:endDate].index[0],  bh_rois[startDate:endDate].index[-1],
                    n_rv, Q, n_samplings, diffobj.n_rules, diffobj.n_periods, pvalue))
                
            
    with open(resFile, 'ab') as fout:
        fout.write(avgIO.getvalue())
    avgIO.close()
    
    print "SPA4BHSymbol SPA, elapsed %.3f secs"%(time.time()-t0)



def SPA4SymbolPortfolioSize(modelType = "fixed", years=None):
    '''
    model for profit test of portflio size 
    '''
    t0 = time.time()
    
    if modelType == "fixed":
        n_rvs = range(5, 55, 5)
        hist_periods = range(50, 130, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95")
        myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", 
                             "LargestMarketValue_200501")
        if not years:
            resFile = os.path.join(ExpResultsDir, "SPA", 
                       "SPA_Fixed_Profit.csv")
        else:
            resFile = os.path.join(ExpResultsDir, "SPA", 
                       "SPA_year_Fixed_Profit.csv")
       
    elif modelType == "dynamic":
        n_rvs = range(5, 55, 5)
        hist_periods = range(90, 120+10, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7")
        myDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", 
                             "LargestMarketValue_200501_rv50")
        if not years:
            resFile = os.path.join(ExpResultsDir, "SPA", 
                               "SPA_Dynamic_Profit.csv")
        else:
            resFile = os.path.join(ExpResultsDir, "SPA", 
                               "SPA_year_Dynamic_Profit.csv")

    #stats file
    avgIO = StringIO()
    if not os.path.exists(resFile):        
        if not years: 
            avgIO.write('n, h, alpha, SPA_Q, sampling, n_rule, n_period, P-value\n')
        else:
            avgIO.write('startDate, endDate, n, h, alpha, SPA_Q, sampling, n_rule, n_period, P-value\n')
      
    
    base_rois = None
    tgt_rois = []
       
    for n_rv in n_rvs:
        t1 = time.time()
        
        #set base ROI
        n_periods = 2236
        base_rois = np.zeros(n_periods)
     
        #load exp ROI
        for period in hist_periods:
            for alpha in alphas:
                if modelType == "fixed":
                    dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                elif modelType == "dynamic":
                    dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                    
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                if len(exps) >3:
                    exps = exps[:3]
                    
                for exp in exps:
                    #load comparison rois
                    df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
                    proc = df.sum(axis=1)
                    exp_finalWealth = proc[-1]
                    if exp_finalWealth >= 0:
                        wrois =  proc.pct_change()
                        wrois[0] = 0
                        tgt_rois.append(wrois)
                    
        print " SPA4Symbol n_rv: %s, load data OK, %.3f secs"%(n_rv, time.time()-t1)
        
        if not years:
            diffobj = ROIDiffObject(base_rois)
            for rois in tgt_rois:
                diffobj.setCompareROIs(rois)
        
            t2 = time.time()
            #SPA test
            Q = 0.5
            n_samplings = 5000
            verbose = True
            pvalue = SPATest.SPATest(diffobj, Q, n_samplings, "SPA_C", verbose)
            print "n_rv:%s, (n_rules, n_periods):(%s, %s), SPA_C:%s elapsed:%.3f secs"%(n_rv,
                        diffobj.n_rules, diffobj.n_periods, pvalue, time.time()-t2)
            avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s\n"%(n_rv, period, alpha, Q, n_samplings, 
                                                     diffobj.n_rules, diffobj.n_periods, pvalue))
        else:
            for year in years:
                startDate, endDate = date(year, 1, 1), date(year, 12, 31)
                dataLen = tgt_rois[0][startDate:endDate]
                
                diffobj = ROIDiffObject(np.zeros(dataLen))
                for rois in tgt_rois:
                    diffobj.setCompareROIs(rois[startDate:endDate])
                    
                t2 = time.time()
                Q = 0.5
                n_samplings = 5000
                verbose = True
                pvalue = SPATest.SPATest(diffobj, Q, n_samplings, "SPA_C", verbose)
                print "year:%s n_rv:%s, (n_rules, n_periods):(%s, %s), SPA_C:%s elapsed:%.3f secs"%(
                        year, n_rv, diffobj.n_rules, diffobj.n_periods, pvalue, time.time()-t2)
                avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(
                   tgt_rois[0][startDate:endDate].index[0],  tgt_rois[0][startDate:endDate].index[-1],
                    n_rv, period, alpha, Q, n_samplings, diffobj.n_rules, diffobj.n_periods, pvalue))
            
        
    with open(resFile, 'ab') as fout:
        fout.write(avgIO.getvalue())
    avgIO.close()
    
    print "SPA4Symbol SPA, elapsed %.3f secs"%(time.time()-t0)


def SPA4Symbol(modelType = "fixed"):
    '''
    model for profit test of each parameter 
    '''
    t0 = time.time()
    
    if modelType == "fixed":
        n_rvs = range(5, 55, 5)
        hist_periods = range(50, 130, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7", 
              "0.75", "0.8", "0.85", "0.9", "0.95")
        myDir = os.path.join(ExpResultsDir, "fixedSymbolSPPortfolio", 
                             "LargestMarketValue_200501")
        
        resFile = os.path.join(ExpResultsDir, "SPA", 
                       "SPA_Fixed_eachParam_Profit.csv")
       
       
    elif modelType == "dynamic":
        n_rvs = range(5, 55, 5)
        hist_periods = range(90, 120+10, 10)
        alphas = ("0.5", "0.55", "0.6", "0.65", "0.7")
        myDir = os.path.join(ExpResultsDir, "dynamicSymbolSPPortfolio", 
                             "LargestMarketValue_200501_rv50")
        
        resFile = os.path.join(ExpResultsDir, "SPA", 
                               "SPA_Dynamic_eachParam_Profit.csv")
     
    avgIO = StringIO()
    if not os.path.exists(resFile):         
        avgIO.write('n, h, alpha, SPA_Q, sampling, n_rule, n_period, P-value\n')
      
    for n_rv in n_rvs:
      
        
        #load exp ROI
        for period in hist_periods:
            for alpha in alphas:
                t1 = time.time()
                if modelType == "fixed":
                    dirName = "fixedSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                elif modelType == "dynamic":
                    dirName = "dynamicSymbolSPPortfolio_n%s_p%s_s200_a%s"%(n_rv, period, alpha)
                    
                exps = glob(os.path.join(myDir, dirName, "20050103-20131231_*"))
                if len(exps) >3:
                    exps = exps[:3]
                
                #set base ROI
                n_periods = 2236
                base_rois = np.zeros(n_periods)
                diffobj = ROIDiffObject(base_rois)
                    
                for exp in exps:
                    #load comparison rois
                    df = pd.read_pickle(os.path.join(exp, 'wealthProcess.pkl'))
                    proc = df.sum(axis=1)
                    exp_finalWealth = proc[-1]
                    if exp_finalWealth >= 0:
                        wrois =  proc.pct_change()
                        wrois[0] = 0
                        diffobj.setCompareROIs(wrois)
                    
                print " SPA4Symbol n-h-alpha:%s-%s-%s, load data OK, %.3f secs"%(n_rv, period, alpha, time.time()-t1)
     
                t2 = time.time()
                #SPA test
                Q = 0.5
                n_samplings = 5000
                verbose = True
                pvalue = SPATest.SPATest(diffobj, Q, n_samplings, "SPA_C", verbose)
                print "n-h-alpha:%s-%s-%s, (n_rules, n_periods):(%s, %s), SPA_C:%s elapsed:%.3f secs"%(
                            n_rv, period, alpha,
                            diffobj.n_rules, diffobj.n_periods, pvalue, time.time()-t2)
                avgIO.write("%s,%s,%s,%s,%s,%s,%s,%s\n"%(n_rv, period, alpha, Q, n_samplings, 
                                                         diffobj.n_rules, diffobj.n_periods, pvalue))
      
    with open(resFile, 'ab') as fout:
        fout.write(avgIO.getvalue())
    avgIO.close()
    
    print "SPA4Symbol SPA, elapsed %.3f secs"%(time.time()-t0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SPTest')

    parser.add_argument('-y', '--years', action='store_true', help="SPA by year")
    args = parser.parse_args()
    
    if not args.years:
        print "full data"
#         SPA4BHSymbol(modelType="fixed")
#         SPA4BHSymbol(modelType="dynamic")
        SPA4Symbol(modelType = "fixed")
        SPA4Symbol(modelType = "dynamic")
    else:
        print "yearly data"
        years = range(2005, 2013+1)
#         SPA4BHSymbol("fixed", years)
#         SPA4BHSymbol("dynamic", years)
#         SPA4Symbol("fixed", years)
#         SPA4Symbol("dynamic", years)
