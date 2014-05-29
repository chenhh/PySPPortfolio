# -*- coding: utf-8 -*-
'''
Created on 2013/10/22

@author: Hung-Hsin Chen
@email: chenhh@par.cse.nsysu.edu.tw

implementation of the following papers:

Halbert White,  “A REALITY CHECK FOR DATA SNOOPING,” Econometrica, 
Vol. 68, No. 5, pp. 1097-1126, 2000.

Romano, J. P. and M. Wolf,.  “Stepwise multiple testing as formalized data 
snooping,” Econometrica, Vol . 73, pp. 1237–1282, 2005.

P.R. Hansen,  “A test for superior predictive ability, ” 
Journal of Business and Economic Statistics, Vol. 23, No. 4, pp. 365-380, 2005.

Po-Hsuan Hsu , Yu-Chin Hsub, Chung-Ming Kuan, “Testing the predictive ability 
of technical analysis using a new stepwise test without data snooping bias,” 
Journal of Empirical Finance, Vol. 17, pp. 471-484, 2010.


'''
from time import time
from __future__ import division
cimport cython
cimport numpy as np
import numpy as np

from multiprocessing import Pool
import multiprocessing as mp
import numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t


np.seterr(all='raise')


def blockBootstrap(diffMtx, length=1, avgType="RC", devDiffColMtx=None):
    '''
    @diffMtx: numpy.matrix, row index: model id, column index: time period
    @length: positive integer, sampling length
    '''
    assert isinstance(diffMtx, np.matrix)
    assert length >= 1
    
    n_models, n_periods = diffMtx.shape
    colidx = np.zeros(n_periods, dtype=np.int)
        
    pts, r = divmod(n_periods, length)
    pts = pts+1 if r != 0 else pts
    
    for pt in xrange(pts):
        t = np.random.randint(0, n_periods)
        idx = pt * length
        for lx in xrange(length):
            loc = idx + lx
            if loc < n_periods:
                colidx[loc] = (t+lx) % n_periods
            else:
                break
  
    samplingMtx = diffMtx[:, colidx]
    avgSamplingColMtx = sampleAverage(diffMtx, samplingMtx, avgType, devDiffColMtx)
    return  avgSamplingColMtx
   
   
def stationaryBootstrap(diffMtx, Q=0.5, avgType="RC", devDiffColMtx=None):
    '''
    @diffMtx: numpy.matrix , row index: model id, column index: time period
    @avgType: string, {RC,  SPA_L, SPA_C, SPA_U}
                       RC for Real check test
                       SPA_L, SPA_C, SPA_U for SPA test
    if Q = 0.5 then mean block size = 1/Q = 2
    
    @devDiffColMtx, numpy.matrix, deviation col vector for SPA test 
    '''
    assert 0 <= Q <= 1
    assert isinstance(diffMtx, np.matrix)

    
    n_models, n_periods = diffMtx.shape
    colidx = np.zeros(n_periods, dtype=np.int)
    colidx[0] = np.random.randint(0, n_periods)
    
    
    for t in xrange(1, n_periods):
        u = np.random.rand()
        if u < Q:
            colidx[t]=np.random.randint(0, n_periods)
        else:
            colidx[t] = colidx[t-1] + 1
            if colidx[t] >= n_periods:
                colidx[t] = 0

    samplingMtx = diffMtx[:, colidx]
    
    avgSamplingColMtx = sampleAverage(diffMtx, samplingMtx, avgType, devDiffColMtx)
    return  avgSamplingColMtx
    

def sampleAverage(diffMtx, samplingMtx, avgType="RC", devDiffColMtx=None):
    '''
    @diffMtx, numpy.matrix, different matrix comparing with benchmark
    @samplingMtx, numpy.matrix, sampling matrix of diffMtx
    @avgType, string, average column vector of diffMtx
    @devDiffColMtx, numpy.matrix, deviation col vector for SPA test 
    '''
    n_models, n_periods = diffMtx.shape
    
    #shape: n_models*1 (model average of sampling)
    if avgType == "RC":
        avgSamplingColMtx = samplingMtx.mean(axis=1)     
    
    elif avgType == "SPA_L":
        #bootstrapMean[k] = sum_t(bootStrapDiff[k][t] -max(0, mean[k]))/T 
        avgColMtx = diffMtx.mean(axis=1)
        diffSamplingMtx = samplingMtx - \
            np.maximum(np.asmatrix(np.zeros((n_models, 1))), avgColMtx)
        
        avgSamplingColMtx = diffSamplingMtx.mean(axis=1)

        diffSamplingMtx = samplingMtx - \
        np.maximum(np.asmatrix(np.zeros((n_models, 1))), avgColMtx)
        
        avgSamplingColMtx = diffSamplingMtx.mean(axis=1)
    
    elif avgType == "SPA_C":
        # bootStrapMean[k] = sum_t(bootStrapDiff[k][t]-
        # mean[k] * indicator(mean[k]>=sqrt(var/n*2*loglogT)))/T

        avgColMtx = diffMtx.mean(axis=1)
        lowerBound = -1. * devDiffColMtx * np.sqrt(
                            2.* np.log(np.log(n_periods))/n_periods)
        diffSamplingMtx = samplingMtx - np.multiply(avgColMtx, 
                                        avgColMtx >= lowerBound )
        avgSamplingColMtx = diffSamplingMtx.mean(axis=1)
   
    elif avgType == "SPA_U":
        #bootstrapMean[k] = sum_t(bootStrapDiff[k][t]-mean[k])/T
        avgColMtx = diffMtx.mean(axis=1)
        diffSamplingMtx =samplingMtx - avgColMtx
        avgSamplingColMtx = diffSamplingMtx.mean(axis=1)
        
    else:
        raise ValueError("unknown average type %s !!"%(avgType))
    
    #n_models * 1
    return  avgSamplingColMtx


class ROIDiffObject(object):
    def __init__(self, baseROI):
        self.baseROI = baseROI
        self.n_periods = baseROI.size
        self.n_rules = 0
        self.ROIMtx = None

    def setROI(self, ROIs):
        ''''''
        assert len(ROIs) == self.n_periods
       
        if self.ROIMtx is None:  
            self.ROIMtx = np.asmatrix(ROIs)
        else: 
            self.ROIMtx = np.vstack((self.ROIMtx, ROIs))
        self.n_rules += 1
    
    def getROIDiffMatrix(self):
        diffMtx = self.ROIMtx - self.baseROI
        return diffMtx


class TradingRuleDiffObject(object):
    '''different matrix of loss function for predictive ability test'''
    
    def __init__(self, dataROI, benchmarkSignals, 
                 transFee = 0.003, shortSell=False):
        '''
        @dataROI: numpy.array or pandas.Series, ROIs for test
        @benchmarkSignals, list of (0, 1, -1), length = n_periods + 1 
                          (containing the previous signal)
        @prevSignals, list of (0, 1, -1), length = n_rules + 1
        @transFee, positive float, transaction fee
        @shortSell: boolean, if false then the signal is 0 or 1, else 
                             the signal may -1, 0 or 1 
        '''
        assert dataROI.size > 0
        self.dataROI = dataROI
        self.n_periods = dataROI.size
        self.n_rules = 0
        
        assert len(benchmarkSignals) == self.n_periods + 1
        if not shortSell:
            assert np.all(s in (0, 1) for s in benchmarkSignals)
        else:
            assert np.all(s in (0, 1, -1) for s in benchmarkSignals)
    
        self.benchmarkSignals = np.asmatrix(benchmarkSignals)
                     
        self.ruleSignalMatrix = None    #n_rules * (n_periods + 1)
        self.transFee = transFee
        self.shortSell = shortSell
        
     
    def setRuleSignal(self, ruleSignals):
        '''
        @ruleSignals: list of 0, 1 and -1, competiting rule signals
        the size of each signal is (n_periods + 1), containing previous signal
        '''
        assert len(ruleSignals) == self.n_periods + 1
        if not self.shortSell:
            assert np.all(signal in (0,1) for signal in ruleSignals)
        else:
            assert np.all(signal in (0, 1, -1) for signal in ruleSignals)
        
        if self.ruleSignalMatrix is None:  
            self.ruleSignalMatrix = np.asmatrix(ruleSignals)
        else: 
            self.ruleSignalMatrix = np.vstack((self.ruleSignalMatrix, 
                                                np.asmatrix(ruleSignals)))
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
        differnece matrix without considering transaction fee
        L[k][t+1] = ROI[t+1]*S[k][t] - ROI[t+1]*S[0][t]
        
        t:           [        0 ,        1 ,        2 ,        3 ]
        shiftedROIs: [      r[0],      r[1],      r[2],      r[3]]
        kth-signals: [   s[prev],      s[0],      s[1],      s[2]]
            
        @return diffMatrix, shape: n_rules * n_periods
        avg(L[k]) > 0, the base model is inferior to kth model
        avg(L[k]) < 0, the base model is superior to kth model
        
        the above definition is consistent with the null hypothesis
        ''' 
        diffMatrix = self.getROIWithTransFeeDiffMatrix()
        return  diffMatrix
    
    def getROIWithTransFeeDiffMatrix(self):
        '''
        differnece matrix with considering transaction fee
        L[k][t+1] = (ROI[t+1]*S[k][t] - abs(S[k][t] - S[k][t-1]) * transFee) - 
                    (ROI[t+1]*S[0][t] - abs(S[0][t] - S[0][t-1]) * transFee) 

        t:           [          0 ,        1 ,        2 ,        3 ]
        shiftedROIs: [        r[0],      r[1],      r[2],      r[3]]
        kth-signals: [     s[prev],      s[0],      s[1],      s[2]]
        diffSignals: [s[0]-s[prev], s[1]-s[0], s[2]-s[1], s[3]-s[2]]
        
        @return diffMatrix, shape: n_rules * n_periods
        avg(L[k]) > 0, the base model is inferior to kth model
        avg(L[k]) < 0, the base model is superior to kth model
        
        the above definition is consistent with the null hypothesis
        '''
        #benchmarkROI, 1 * n_periods
        diffBenchmarkSignals = np.diff(self.benchmarkSignals)
        bTransFees = np.abs(diffBenchmarkSignals) * self.transFee        
        benchmarkROI = np.multiply(self.dataROI, self.benchmarkSignals[0, :-1]) - bTransFees
        
        #rulesROI, n_rules * n_periods
        diffRuleSignalMatrix = np.diff(self.ruleSignalMatrix)
        rTransFees = np.abs(diffRuleSignalMatrix) * self.transFee
        rulesROI = np.multiply(self.dataROI, self.ruleSignalMatrix[:, -1]) - rTransFees
        
        diffMatrix = rulesROI - benchmarkROI
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
    

        
def RCTest(diffObj, Q=0.5, n_samplings=1000, verbose=False):
    '''
    @diffObj, different object
    @Q: float in [0, 1), parameter of stationary bootstrap
    @n_samplings: positive integer, number of sampling
    '''
    #diffMtx shape: n_rules * n_periods
    assert diffObj.n_rules > 0 and n_samplings >= 100
    diffMtx = diffObj.getROIDiffMatrix()        
    
    #RC statistic
    sqrtPeriods = np.sqrt(diffObj.n_periods)
    avgDiffColMtx = diffMtx.mean(axis=1)
    dataStatistic = np.max(sqrtPeriods*avgDiffColMtx) #scalar
    
    #bootstrap sampling
    t1 = time()
    pool = Pool(mp.cpu_count())
    results = [pool.apply_async(stationaryBootstrap, 
                   (diffMtx, Q))
                   for _ in xrange(n_samplings)]
     
    [result.wait() for result in results]
    avgDiffSamplingColMtxs = [result.get() for result in results]
    pool.close()
    pool.join()
       
    if verbose:
        print "RC test, sampling %s used %.3f secs"%(n_samplings, time()-t1)
    
    #RC sampling statistics
    samplingStatistics = (np.max( sqrtPeriods*(avgDiffSampleColMtx-avgDiffColMtx) )
                                for avgDiffSampleColMtx in avgDiffSamplingColMtxs)

    loses = np.sum(1. for statistic in samplingStatistics if dataStatistic < statistic)
    pvalue = loses/n_samplings
    
    if verbose:
        print "RCTest, loses:{0}/{1}, pvalue:{2}".format(loses, n_samplings, pvalue)
    
    return pvalue


def SPATest(diffObj, Q=0.5, n_samplings=1000, avgType="SPA_C", verbose=False):
    '''
    @diffObj, different object
    @Q: float in [0, 1), parameter of stationary bootstrap
    @n_samplings: positive integer, number of sampling
    @avgType: string, {SPA_L, SPA_C, or SPA_U}
    @return p-value
    '''
    assert diffObj.n_rules > 0 and n_samplings >= 100
    
    diffMtx = diffObj.getROIDiffMatrix()
    #shape: n_rules*1
    avgDiffColMtx = diffMtx.mean(axis=1)          
    devDiffColMtx = diffObj.getDeviation(diffMtx, Q)
    
    
    #SPA data statistic (dev can not be 0)
    sqrtPeriods = np.sqrt(diffObj.n_periods)
    dataStatistic = np.max(np.max(sqrtPeriods * np.divide(avgDiffColMtx,devDiffColMtx), 0.))
    
    #3 types of adjusted mean of bootstrap, type L, type C, type U 
    t1 = time()
    pool = Pool(mp.cpu_count())
    results = [pool.apply_async(stationaryBootstrap, 
                (diffMtx, Q, avgType, devDiffColMtx))
                for _ in xrange(n_samplings)]
    [result.wait() for result in results]
      
    avgDiffSampleColVecs = [result.get() for result in results]
    pool.close()
    pool.join()

    if verbose:
        print "SPA test: %s sampling %s used %.3f secs"%(avgType,
                                    n_samplings, time()-t1)
    
    #SPA sampling statistics(the same deviation as data)
    sampleStatistics = ( np.max(np.max( sqrtPeriods * np.divide(avgColVec, devDiffColMtx), 0) ) 
                        for avgColVec in avgDiffSampleColVecs)
    
    loses = np.sum(1. for statistic in sampleStatistics if dataStatistic < statistic)
    pvalue = loses/n_samplings
    if verbose:
        print "SPA Test %s loses:%s/%s, pvalue:%s"%(avgType, loses, n_samplings, pvalue)
    
    return pvalue

  


def _filterSignificantIDs(diffMtx, noneSignificantIDs):
    '''
    filtering significant model id in the diffMtx
    and preserving none Significant in the diffMtx
    
    @nonsignificantIDs: list of none significant indices
    @return filtered diffMtx
    '''
    n_rules = diffMtx.shape[0]
    #first round
    if len(noneSignificantIDs) == n_rules:
        #all models are not significant
        return diffMtx
    else:
        assert len(noneSignificantIDs) > 0
        return diffMtx[noneSignificantIDs, :]


def StepwiseRCTest(diffObj, Q=0.5, n_samplings=1000,  alpha=0.05, verbose=False):
    '''
    @diffObj, different object
    @Q: float in [0, 1), parameter of stationary bootstrap
    @n_samplings: positive integer, number of sampling
    @alpha, positive float: (1-alpha) is the significance
    '''
    assert 0 <alpha <= 0.5
    assert diffObj.n_rules > 0
    
    n_steps = 0
    significant = True
    significantIDs = []
    sqrtPeriods = np.sqrt(diffObj.n_periods)
    
    noneSignificantIDs = range(diffObj.n_rules)
    origDiffMtx = diffObj.getROIDiffMatrix()
    origAvgDiffColMtx = origDiffMtx.mean(axis=1)
     
    while significant:
        if len(noneSignificantIDs) == 0:
            break
        n_steps += 1
        diffMtx = _filterSignificantIDs(origDiffMtx, noneSignificantIDs)
        avgDiffColMtx = diffMtx.mean(axis=1)    
     
        #RC data statistic
        statsColMtx = sqrtPeriods*avgDiffColMtx
        dataStatistic = np.max(statsColMtx)
         
        t1 = time()
        pool = Pool(mp.cpu_count())
        results = [pool.apply_async(stationaryBootstrap, (diffMtx, Q))
                   for _ in xrange(n_samplings)]
        [result.wait() for result in results]
        avgDiffSamplingColMtxs = [result.get() for result in results]
        pool.close()
        pool.join()
        if verbose:
            print "RC stepwise test, step: %s, sampling %s, elapsed %.3f secs"%(
                                        n_steps, n_samplings, time()-t1)
         
        #RC sampling statistics
        samplingStatistics = np.array([np.max(sqrtPeriods*(avgDiffSampleColMtx-avgDiffColMtx)) 
                                for avgDiffSampleColMtx in avgDiffSamplingColMtxs])
         
        loses = (dataStatistic < samplingStatistics).sum()
        pvalue = float(loses)/n_samplings
         
        if n_steps == 1:
            firstPvalue = pvalue
        
        if verbose:
            print "RCStepwiseTest step:%s, loses:%s/%s, pvalue:%s, alpha:%s"%(
                         n_steps, loses, n_samplings, pvalue, alpha)
         
        if pvalue > alpha:
            #no significant model, exit while loop
            significant = False
        else:
            samplingStatistics.sort()   #ascending
            criticalValue =  samplingStatistics[np.ceil(n_samplings*(1.-alpha))]
            ids = [noneSignID for idx, noneSignID in enumerate(noneSignificantIDs) \
                     if float(statsColMtx[idx]) >= criticalValue]
            significantIDs.extend(ids)
            [noneSignificantIDs.remove(signid) for signid in ids]
             
        if verbose:
            print "step %s, significantIDs:%s"%(n_steps, significantIDs)
           
    if verbose:
        for nid in xrange(diffObj.n_rules):
            print "[%s], avgDiff:%s"%(nid, origAvgDiffColMtx[nid])
        
        for sid in significantIDs:
            print sid, "avgDiff: ",origAvgDiffColMtx[sid]
        print "RCStepwiseTest, rules: %s, ids: %s"%(len(significantIDs), 
                                                    significantIDs)
    
    return firstPvalue, significantIDs, n_steps
    

def StepwiseSPATest(diffObj, Q=0.5, n_samplings=1000,  alpha=0.05, avgType="SPA_C", verbose=False):
    '''
    @diffObj, different object
    @Q: float in [0, 1), parameter of stationary bootstrap
    @n_samplings: positive integer, number of sampling
    @avgType: string, {SPA_L, SPA_C, or SPA_U}
    @alpha, positive float: (1-alpha) is the significance
    '''
    assert 0< alpha<=0.5
    assert diffObj.n_rules
        
    n_steps = 0
    significant = True
    noneSignificantIDs = range(diffObj.n_rules)
    significantIDs = []
    sqrtPeriods = np.sqrt(diffObj.n_periods)
    origDiffMtx = diffObj.getROIDiffMatrix()
    origAvgDiffColMtx  =  origDiffMtx.mean(axis=1)
          
    while significant:
        if len(noneSignificantIDs) == 0:
            break
        n_steps += 1
#         print "diffObj:%s, steps:%s, ids:%s"%(diffObj.n_rules, n_steps, noneSignificantIDs)
#         print "diffROIs: %s"%(diffObj.getROIDiffMatrix().sum(axis=1))
        diffMtx = _filterSignificantIDs(origDiffMtx, noneSignificantIDs)
        avgDiffColMtx = diffMtx.mean(axis=1)    
        devDiffColMtx = diffObj.getDeviation(diffMtx, Q)
        
        #SPA data statistic
        statsColMtx = sqrtPeriods * np.divide(avgDiffColMtx, devDiffColMtx)
        dataStatistic = np.max(np.max(statsColMtx), 0.0)
       
        t1 = time()
        pool = Pool(mp.cpu_count()-1)
        results = [pool.apply_async(stationaryBootstrap, 
                    (diffMtx, Q, avgType, devDiffColMtx)) 
                   for _ in xrange(n_samplings)]
        [result.wait() for result in results]
        avgDiffSampleColMtxs = [result.get() for result in results]
        pool.close()
        pool.join()
        if verbose:
            print "SPA stepwise test, steps: %s, sampling: %s, elapsed %.3f secs"%(
                        n_steps, n_samplings, time()-t1)
          
        #SPA sampling statistics
        samplingStatistics = np.array([np.max(np.max(sqrtPeriods* np.divide(avgColMtx, devDiffColMtx), 0.) ) 
                                             for avgColMtx in avgDiffSampleColMtxs])
        
        loses = (dataStatistic < samplingStatistics).sum()
        pvalue = float(loses)/n_samplings
        
        if n_steps == 1:
            firstPvalue = pvalue
        
        if verbose:
            print "SPAStepwiseTest, step:%s, loses:%s/%s, pvalue:%s, alpha:%s"%(
                    n_steps, loses, n_samplings, pvalue, alpha)
          
        if pvalue > alpha:
            #no significant model, exit while loop
            significant = False
        else:
            samplingStatistics.sort()   #ascending
            criticalValue = samplingStatistics[int(n_samplings*(1-alpha))]
            ids = [noneSignID for pt, noneSignID in enumerate(noneSignificantIDs) 
                     if float(statsColMtx[pt]) >= criticalValue]
            significantIDs.extend(ids)
            [noneSignificantIDs.remove(signid) for signid in ids]
             
        if verbose:
            print "step %s, significantIDs:%s"%(n_steps, significantIDs)
      
              
    if verbose:
        for nid in xrange(diffObj.n_rules):
            print "[%s], avgDiff:%s"%(nid, origAvgDiffColMtx[nid])
        
        for sid in significantIDs:
            print "significant:", sid, "avgDiff: ",origAvgDiffColMtx[sid]
        print "SPAStepwiseTest, rules: %s, ids: %s"%(len(significantIDs), significantIDs)
    
    return firstPvalue, significantIDs, n_steps
     

def testSPA():
    n_rules = 10
    n_periods = 20
    n_samplings = 100
    Q = 0.5
    alpha = 0.2
    verbose=True
    
    ROIs = np.random.randn(n_periods)
    baseSignals = np.zeros(n_periods+1)
    
    diffObj = TradingRuleDiffObject(ROIs,baseSignals)
    
    for _ in xrange(n_rules):
        tradingSignals = np.random.randint(0, 2, n_periods+1)
        tradingSignals[-1] = 0
        diffObj.setRuleSignal(tradingSignals)
   
    print "RC:", RCTest(diffObj, Q, n_samplings, verbose)
    print "SPA_L:", SPATest(diffObj, Q, n_samplings, "SPA_L", verbose)
    print "SPA_C:", SPATest(diffObj, Q, n_samplings, "SPA_C", verbose)
    print "SPA_U:", SPATest(diffObj, Q, n_samplings, "SPA_U", verbose)
    print "stepwise RC:", StepwiseRCTest(diffObj, Q, n_samplings,  alpha, verbose)
    print "stepwise SPA_C:", StepwiseSPATest(diffObj, Q, n_samplings,  alpha, "SPA_C", verbose)


if __name__ == '__main__':
    testSPA()