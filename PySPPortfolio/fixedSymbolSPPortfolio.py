# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
start from 2005/1/1 to 2013/12/31
'''

from __future__ import division
import argparse
import os
import platform
import time
from datetime import date
import numpy as np
import numpy.linalg as la 
import scipy.stats as spstats
import pandas as pd
from HKW_wrapper import HKW_wrapper
from MinCVaRPortfolioSP import MinCVaRPortfolioSP
import simplejson as json 

FileDir = os.path.abspath(os.path.curdir)
PklBasicFeaturesDir = os.path.join(FileDir,'pkl', 'BasicFeatures')
if platform.uname()[0] == 'Linux':
    ExpResultsDir =  os.path.join('/', 'home', 'chenhh' , 'Dropbox', 
                                  'financial_experiment', 'PySPPortfolio')
    
elif platform.uname()[0] =='Windows':
    ExpResultsDir= os.path.join('C:\\', 'Dropbox', 'financial_experiment', 
                                'PySPPortfolio')    
    
def constructModelMtx(symbols, startDate=date(2005,1,1), endDate=date(2013,12,31), 
                      money=1e6, hist_period=60, buyTransFee=0.001425, 
                      sellTransFee=0.004425, alpha = 0.95, debug=False):
    '''
    -注意因為最後一期只結算不買賣
    -DataFrame以Date取資料時，有包含last day, 即df[startDate: endDate]
    -包含了endDate的資料，但是使用index取資料時，如df[2:10]，則不包含df.ix[10]的資料
    @param symbols, list
    @param startDate, endDate, datetime.date
    @param momey, positive float
    @param hist_period, positive integer
    
    dataFrame取單一row用df.ix[idx]
    df.index.get_loc(startDate)找index所在位置
    @return riskyRetMtx, numpy.array, size: (n_rv * hist_period+T)
    ''' 
    
    #read data
    dfs = []
    transDates = None
    for symbol in symbols:
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        tmp = df[startDate: endDate]
        startIdx = df.index.get_loc(tmp.index[0])
        endIdx =  df.index.get_loc(tmp.index[-1])
        if startIdx < (hist_period-1):
            raise ValueError('%s do not have enough data'%(symbol))
        #index from [0, hist_period-1] for estimating statistics
        data = df[startIdx-hist_period+1: endIdx+1]
       
        #check all data have the same transDate
        if transDates is None:
            transDates = data.index.values
        if not np.all(transDates == data.index.values):
            raise ValueError('symbol %s do not have the same trans. dates'%(symbol))
        dfs.append(data)
    
    #fixed transDate data
    fullTransDates = transDates             #size: hist_period + (T +1) -1
    transDates = transDates[hist_period-1:] #size: T+1
    
    #最後一期只結算不買賣, 所以要減一期 
    n_rv, T = len(symbols), len(transDates) - 1
    allRiskyRetMtx = np.empty((n_rv, hist_period+T))
    for idx, df in enumerate(dfs):
        allRiskyRetMtx[idx, :] = df['adjROI'].values/100.
    
    riskFreeRetVec = np.zeros(T+1)
    buyTransFeeMtx = np.ones((n_rv, T)) * buyTransFee
    sellTransFeeMtx = np.ones((n_rv, T))* sellTransFee
    
    #allocated 為已配置在risky asset的金額
    allocatedWealth = np.zeros(n_rv)
    depositWealth = money
    
    if debug:
        print "n_rv: %s, T: %s, hist_period: %s"%(n_rv, T, hist_period)
        print "allRiskyRetMtx size (n_rv * hist_period+T): ", allRiskyRetMtx.shape
        print "riskFreeRetVec size (T+1): ", riskFreeRetVec.shape
        print "buyTransFeeMtx size (n_rv, T): ",  buyTransFeeMtx.shape
        print "sellTransFeeMtx size (n_rv, T): ", sellTransFeeMtx.shape
        print "allocatedWealth size n_rv: ",  allocatedWealth.shape
        print "depositWealth value:", depositWealth
        print "transDates size (T+1):", transDates.shape
        print "full transDates, size(hist+T): ", fullTransDates.shape
        print "alpha value:", alpha
    
    return {
        "n_rv": n_rv,
        "T": T,
        "allRiskyRetMtx": allRiskyRetMtx, #size: n_rv * (hist_period+T)
        "riskFreeRetVec": riskFreeRetVec,   #size: T+1
        "buyTransFeeMtx": buyTransFeeMtx,   #size: n_rv * T
        "sellTransFeeMtx": sellTransFeeMtx, #size: n_rv * T
        "allocatedWealth": allocatedWealth, #size: n_rv
        "depositWealth": depositWealth,     #size: 1 
        "transDates": transDates,           #size: (T+1)
        "fullTransDates": fullTransDates,   #size: (hist_period+T)
        "alpha": alpha                      #size：１
        }


def fixedSymbolSPPortfolio(symbols, startDate, endDate,  money=1e6,
                           hist_period=20, n_scenario=1000,
                           buyTransFee=0.001425, sellTransFee=0.004425,
                           alpha=0.95, scenFunc="Moment", solver="glpk", 
                           debug=False):
    '''
    -固定投資標的物(symbols)，只考慮buy, sell的交易策略
    -假設symbols有n_rv個，投資期數共T期(最後一期不買賣，只結算)
    
    @param symbols, list, target assets
    @param startDate, endDate, datetime.date, 交易的起始，結束日期
    @param money, positive float, 初使資金
    @param hist_period, positive integer, 用於計算moment與corr mtx的歷史資料長度
    @param n_scenario, positive integer, 每一期產生的scenario個數
    @param buyTransFee, sellTransFee, float, 買進與賣出手續費
    @param alpha, float, confidence level of the CVaR
    @scenFunc, string, 產生scenario的function
    @solver, string, 解stochastic programming的solver
    
    @return {
        "n_rv": n_rv,
        "T": T,
        "allRiskyRetMtx": allRiskyRetMtx,   #size: n_rv * (hist_period+T)
        #[0:hist_period]用於估計moments與corrMtx
        "riskFreeRetVec": riskFreeRetVec,   #size: T+1
        "buyTransFeeMtx": buyTransFeeMtx,   #size: n_rv * T
        "sellTransFeeMtx": sellTransFeeMtx, #size: n_rv * T
        "allocatedWealth": allocatedWealth, #size: n_rv
        "depositWealth": depositWealth,     #size: 1 
        "transDates": transDates,           #size: (T+1)
        "fullTransDates": fullTransDates,   #size: (hist_period+T)
         "alpha": alpha                      #size：１
        }
    '''
    
    t0 = time.time()
    param = constructModelMtx(symbols, startDate, endDate, money, hist_period,
                              buyTransFee, sellTransFee, alpha, debug)
    print "constructModelMtx %.3f secs"%(time.time()-t0)
    
    n_rv, T =param['n_rv'], param['T']
    allRiskyRetMtx = param['allRiskyRetMtx']
    riskFreeRetVec = param['riskFreeRetVec']
    buyTransFeeMtx = param['buyTransFeeMtx']
    sellTransFeeMtx = param['sellTransFeeMtx']
    allocatedWealth = param['allocatedWealth']
    depositWealth = param['depositWealth']
    transDates = param['transDates']
    fullTransDates = param['fullTransDates']
    
    #process from t=0 to t=(T+1)
    buyProcess = np.zeros((n_rv, T))
    sellProcess = np.zeros((n_rv, T))
    wealthProcess = np.zeros((n_rv, T+1))
    depositProcess = np.zeros(T+1)
    VaRProcess = np.zeros(T)
    
    genScenErrDates = []
    for tdx in xrange(T):
        tloop = time.time()
        transDate = pd.to_datetime(transDates[tdx])
         
        #投資時已知當日的ret(即已經知道當日收盤價)
        t = time.time()
        subRiskyRetMtx = allRiskyRetMtx[:, tdx:(hist_period+tdx)]

        if scenFunc == "Moment":
            moments = np.empty((n_rv, 4))
            moments[:, 0] = subRiskyRetMtx.mean(axis=1)
            moments[:, 1] = subRiskyRetMtx.std(axis=1)
            moments[:, 2] = spstats.skew(subRiskyRetMtx, axis=1)
            moments[:, 3] = spstats.kurtosis(subRiskyRetMtx, axis=1)
            corrMtx = np.corrcoef(subRiskyRetMtx)
            MaxTrial = 50
            HKW_MaxIter =50
            MaxErrMom = 1e-1
            MaxErrCorr = 1e-1
            
            print "mom:", moments
            print "corr:", corrMtx
            for kdx in xrange(3):
                MaxErrMom, MaxErrCorr = 1e-3 * (10**kdx), 1e-3 * (10**kdx)
                print "kdx: %s, momErr: %s, corrErr:%s"%(kdx, MaxErrMom, MaxErrCorr)
                scenMtx, rc = HKW_wrapper.HeuristicMomentMatching(moments, corrMtx, n_scenario,
                                            MaxTrial, HKW_MaxIter, MaxErrMom, MaxErrCorr)
        else:
            raise ValueError("unknown scenFunc %s"%(scenFunc))
        
        print "%s-%s - generate scen. mtx, %.3f secs"%(transDate, scenFunc, time.time()-t)
        
        if rc == 0:
            #solve SP
            t = time.time()
            riskyRet = allRiskyRetMtx[:, hist_period+tdx]
            riskFreeRet = riskFreeRetVec[tdx]
            buyTransFee = buyTransFeeMtx[:, tdx]
            sellTransFee =  sellTransFeeMtx[:, tdx]
            predictRiskyRet = scenMtx
            predictRiskFreeRet = 0
            results = MinCVaRPortfolioSP(symbols, riskyRet, riskFreeRet, allocatedWealth,
                           depositWealth, buyTransFee, sellTransFee, alpha,
                           predictRiskyRet, predictRiskFreeRet, n_scenario, 
                           probs=None, solver=solver)
            print "%s - %s solve SP, %.3f secs"%(transDate, solver, time.time()-t)
        else:
            genScenErrDates.append(transDate)
            results = None
        
        #realized today return
        allocatedWealth = allocatedWealth * (1+allRiskyRetMtx[:, hist_period+tdx])
        depositWealth =  depositWealth * (1+riskFreeRetVec[tdx])
        
        if rc == 0 and results is not None:
            #buy action
            for idx, value in enumerate(results['buys']):
                allocatedWealth[idx] += value
                buy = (1 + buyTransFeeMtx[idx, tdx]) * value
                buyProcess[idx, tdx] = buy
                depositWealth -= buy
            
            #sell action
            for jdx, value in enumerate(results['sells']):
                allocatedWealth[idx] -= value
                sell = (1 - sellTransFeeMtx[idx, tdx]) * value
                sellProcess[idx, tdx] = sell
                depositWealth += sell
    
        #log wealth and signal process
        wealthProcess[:, tdx] = allocatedWealth
        depositProcess[tdx] = depositWealth
                                    
        print '*'*75
        print '''%s-%s n%s-h%s-s%s-a%s --scenFunc %s --solver %s, genscenErr:[%s]
                  transDate %s fixed CVaR SP OK, current wealth %s, %.3f secs
               '''%( startDate, endDate, n_rv, hist_period, n_scenario, alpha, scenFunc, solver, 
                len(genScenErrDates),  transDate,  allocatedWealth.sum() + depositWealth,
                time.time()-tloop)
        print '*'*75
    
    #最後一期只結算不買賣
    wealthProcess[:, -1] = allocatedWealth * (1+allRiskyRetMtx[:, -1])
    depositProcess[-1] =  depositWealth * (1+riskFreeRetVec[-1])
   
    finalWealth = (np.dot(allocatedWealth, (1+allRiskyRetMtx[:, -1])) + 
                   depositWealth * (1+riskFreeRetVec[-1]))
    print "final wealth %s"%(finalWealth)
    
    #setup result directory
    t1 = pd.to_datetime(transDates[0]).strftime("%Y%m%d")
    t2 = pd.to_datetime(transDates[-1]).strftime("%Y%m%d")
    rnd = time.strftime("%y%m%d%H%M%S")
    layer1Dir =  "%s_n%s_h%s_s%s_a%s"%(fixedSymbolSPPortfolio.__name__, n_rv, 
                                       hist_period, n_scenario, alpha)
    layer2Dir = "%s-%s_%s"%(t1, t2, rnd)
    resultDir = os.path.join(ExpResultsDir, layer1Dir, layer2Dir)
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)
    
    #store data in pkl
    pd_buyProc = pd.DataFrame(buyProcess.T, index=transDates[:-1], columns=symbols)
    pd_sellProc = pd.DataFrame(sellProcess.T, index=transDates[:-1], columns=symbols) 
    pd_wealthProc = pd.DataFrame(wealthProcess.T, index=transDates, columns=symbols)
    pd_depositProc = pd.Series(depositProcess.T, index=transDates)
    pd_VaRProc = pd.Series(VaRProcess.T, index=transDates[:-1])
    
    records = {
        "buyProcess": pd_buyProc, 
        "sellProcess": pd_sellProc, 
        "wealthProcess": pd_wealthProc, 
        "depositProcess": pd_depositProc, 
        "VaRProcess": pd_VaRProc
    }
    
    #save pkl
    for name, df in records.items():
        pklFileName = os.path.join(resultDir, "%s.pkl"%(name))
        df.save(pklFileName)
   
        csvFileName = os.path.join(resultDir, "%s.csv"%(name))
        df.to_csv(csvFileName)
    
    #generating summary files
    summary = {"n_rv": n_rv,
               "T": T,
               "scenario": n_scenario,
               "alpha": alpha,
               "symbols":  ",".join(symbols),
               "transDates": transDates,    #(T+!)
               "hist_period": hist_period,
               "final_wealth": finalWealth,
               "scenFunc": scenFunc,
               "scen_err_cnt":len(genScenErrDates),
               "scen_err_dates": genScenErrDates,
               "buyProcess": buyProcess,
               "sellProcess": sellProcess,
               "wealthProcess": wealthProcess,
               "depositProcess": depositProcess,
               "VaRProcess": VaRProcess,
               "machine": platform.node(),
               "elapsed": time.time()-t0
               }
    
    fileName = os.path.join(resultDir, 'summary.json')
    with open (fileName, 'w') as fout:
        json.dump(summary, fout)
    
    print "%s-%s n%s-h%s-s%s-a%s --scenFunc %s --solver %s\nsimulation ok, %.3f secs"%(
             startDate, endDate, n_rv, hist_period, n_scenario, alpha,
             scenFunc, solver, time.time()-t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fixedSymbolSPPortfolio')
#     parser.parse_args()
    parser.add_argument('-n', '--symbols', type=int, default=5, help="num. of symbols")
    parser.add_argument('-p', '--histPeriod', type=int, default=40, help="historical period")
    parser.add_argument('-s', '--scenario', type=int, default=200, help="num. of scenario")
    parser.add_argument('-a', '--alpha', type=float, default=0.95, help="confidence level of CVaR")
    parser.add_argument('--solver', choices=["glpk", "cplex"], default="glpk", help="solver for SP")
    parser.add_argument('--scenFunc', choices=["Moment", "Copula"], default="Moment", help="function for generating scenario")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-y', '--year', type=int, choices=range(2005, 2013+1), help="experiment in year")
    group.add_argument('-f', '--full', help="from 2005~2013")
    args = parser.parse_args()

    # 把參數 number 的值印出來
    print args
        
    #market value top 20 (2013/12/31)
    symbols = ['2330', '2317', '6505', '2412', '2454',
                '2882', '1303', '1301', '1326', '2881',
                '2002', '2308', '3045', '2886', '2891',
                '1216', '2382', '2105', '2311', '2912'
               ]
    symbols = symbols[:args.symbols]
    
    if args.year:
        startDate = date(args.year, 2, 1)
        endDate = date(args.year, 3, 31)
        
    money = 1e6
    hist_period = args.histPeriod
    n_scenario = args.scenario
    buyTransFee=0.001425
    sellTransFee=0.004425
    alpha = args.alpha
    scenFunc = args.scenFunc
    solver = args.solver
    debug = False
    fixedSymbolSPPortfolio(symbols, startDate, endDate,  money,
                           hist_period, n_scenario,
                           buyTransFee, sellTransFee,
                           alpha, scenFunc, solver, debug)