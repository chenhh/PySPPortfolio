# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
start from 2005/1/1 to 2013/12/31
there are totally M stocks, choose the best n stocks to invest by 
stochastic integer programming.

there are three parameters (n, h, alpha), whichare stocks n, 
hist. period h and a confidence level alpha,
computing the final wealth

'''

from __future__ import division
import sys
import os
ProjectDir = os.path.join(os.path.abspath(os.path.curdir), '..')
sys.path.insert(0, ProjectDir)

import argparse
import platform
import time
from datetime import date
import numpy as np
import scipy.stats as spstats
import pandas as pd
from scenario.CMoment import HeuristicMomentMatching
from riskOpt.MinCVaRPortfolioSP import MinCVaRPortfolioSIP
import simplejson as json 
from cStringIO import StringIO
from PySPPortfolio import (PklBasicFeaturesDir,  ExpResultsDir)
from fixedSymbolSPPortfolio import constructModelMtx


def dynamicSymbolSPPortfolio(symbols, startDate=date(2005,1,1), 
                        endDate=date(2013,12,31),  money=1e6, 
                        hist_period=20, n_scenario=200,
                        buyTransFee=0.001425, sellTransFee=0.004425,
                        alpha=0.95, scenFunc="Moment", solver="cplex", 
                        n_stock=3, save_pkl=False, save_csv=True, debug=False):
    '''
    -在symbols中最多選n_stcok隻股票投資
    
    @param symbols, list, target assets
    @param startDate, endDate, datetime.date, 交易的起始，結束日期
    @param money, positive float, 初使資金
    @param hist_period, positive integer, 用於計算moment與corr mtx的歷史資料長度
    @param n_scenario, positive integer, 每一期產生的scenario個數
    
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
    assert len(symbols) >= n_stock
    
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
    CVaRProcess = np.zeros(T)
    
    genScenErrDates = []
    scenErrStringIO = StringIO()
    
    for tdx in xrange(T):
        tloop = time.time()
        transDate = pd.to_datetime(transDates[tdx]).strftime("%Y%m%d")
        
        #投資時已知當日的ret(即已經知道當日收盤價)
        t = time.time()
        subRiskyRetMtx = allRiskyRetMtx[:, tdx:(hist_period+tdx)]
        assert subRiskyRetMtx.shape[1] == hist_period
        
        if scenFunc == "Moment":
            moments = np.empty((n_rv, 4))
            moments[:, 0] = subRiskyRetMtx.mean(axis=1)
            moments[:, 1] = subRiskyRetMtx.std(axis=1)
            moments[:, 2] = spstats.skew(subRiskyRetMtx, axis=1)
            moments[:, 3] = spstats.kurtosis(subRiskyRetMtx, axis=1)
            corrMtx = np.corrcoef(subRiskyRetMtx)
            
            converged = False
            for order in xrange(-3, 0): 
                MaxErrMom, MaxErrCorr=10**(order), 10**(order)
                try:
                    scenMtx = HeuristicMomentMatching(moments, corrMtx, 
                                    n_scenario, MaxErrMom, MaxErrCorr)                  
                except ValueError as e:
                    print e
                    scenErrStringIO.write("%s: %s\n"%(transDate, e))
                else:
                    converged = True
                    break
        else:
            raise ValueError("unknown scenFunc %s"%(scenFunc))
        
        print "%s-%s - generate scen. mtx, %.3f secs"%(transDate, scenFunc, time.time()-t)
        
        if converged:
            #successful generating scenarios, solve SP
            t = time.time()
            riskyRet = allRiskyRetMtx[:, hist_period+tdx]
            riskFreeRet = riskFreeRetVec[tdx]
            buyTransFee = buyTransFeeMtx[:, tdx]
            sellTransFee =  sellTransFeeMtx[:, tdx]
            predictRiskyRet = scenMtx
            predictRiskFreeRet = 0
            results = MinCVaRPortfolioSIP(symbols, riskyRet, riskFreeRet, allocatedWealth,
                           depositWealth, buyTransFee, sellTransFee, alpha,
                           predictRiskyRet, predictRiskFreeRet, n_scenario, 
                           probs=None, solver=solver, n_stock=n_stock)
            
            VaRProcess[tdx] = results['VaR']
            CVaRProcess[tdx] = results['CVaR']
            print "%s - %s solve SP, %.3f secs"%(transDate, solver, time.time()-t)
        else:
            #failed generating scenarios
            genScenErrDates.append(transDate)
            results = None
            
        #realized today return
        allocatedWealth = allocatedWealth * (1+allRiskyRetMtx[:, hist_period+tdx])
        depositWealth =  depositWealth * (1+riskFreeRetVec[tdx])
        
        if converged and results is not None:
            #buy action
            for idx, value in enumerate(results['buys']):
                allocatedWealth[idx] += value
                buy = (1 + buyTransFeeMtx[idx, tdx]) * value
                buyProcess[idx, tdx] = buy
                depositWealth -= buy
            
            #sell action
            for idx, value in enumerate(results['sells']):
                allocatedWealth[idx] -= value
                sell = (1 - sellTransFeeMtx[idx, tdx]) * value
                sellProcess[idx, tdx] = sell
                depositWealth += sell
    
        #log wealth and signal process
        wealthProcess[:, tdx] = allocatedWealth
        depositProcess[tdx] = depositWealth
                                    
        print '*'*80
        trainDates = [pd.to_datetime(fullTransDates[tdx]).strftime("%Y%m%d"), 
                      pd.to_datetime(fullTransDates[hist_period+tdx-1]).strftime("%Y%m%d")]
        
        print ' dynamicSymbolSPPortfolio %s-%s n%s-p%s-s%s-a%s --scenFunc %s --solver %s, genscenErr:[%s]'%(
            startDate, endDate, n_stock, hist_period, n_scenario, alpha, 
            scenFunc, solver, len(genScenErrDates))
        
        print 'transDate %s (train:%s-%s) fixed CVaR SP OK, current wealth %s, %.3f secs'%(
                transDate, trainDates[0], trainDates[1], 
                allocatedWealth.sum() + depositWealth, time.time()-tloop)
        print '*'*80
        #end of for
    
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
    layer0Dir =  "%s"%(dynamicSymbolSPPortfolio.__name__)
    layer1Dir =  "LargestMarketValue_200501"
    layer2Dir =  "%s_n%s_p%s_s%s_a%s"%(dynamicSymbolSPPortfolio.__name__, n_stock, 
                                       hist_period, n_scenario, alpha)
    layer3Dir = "%s-%s_%s"%(t1, t2, rnd)
    resultDir = os.path.join(ExpResultsDir,  layer0Dir, layer1Dir, 
                             layer2Dir, layer3Dir)
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)
    
    #store data in pkl
    df_buyProc = pd.DataFrame(buyProcess.T, index=transDates[:-1], 
                              columns=["%s_buy"%(sym) for sym in symbols])
    
    df_sellProc = pd.DataFrame(sellProcess.T, index=transDates[:-1], 
                               columns=["%s_sell"%(sym) for sym in symbols])
    
    df_action = pd.merge(df_buyProc, df_sellProc, left_index=True, right_index=True) 
    
    df_wealth = pd.DataFrame(wealthProcess.T, index=transDates, columns=symbols)
    deposits = pd.Series(depositProcess.T, index=transDates)
    df_wealth['deposit'] = deposits 
   
    df_risk = pd.DataFrame({"VaR": pd.Series(VaRProcess.T, index=transDates[:-1]),
                            "CVaR":  pd.Series(CVaRProcess.T, index=transDates[:-1])
                            }) 
    
    records = { 
        "actionProcess": df_action,
        "wealthProcess": df_wealth, 
        "riskProcess": df_risk
    }
    
    #save csv (don't save to pkl to save disk space)
    for name, df in records.items():
        if save_pkl:
            pklFileName = os.path.join(resultDir, "%s.pkl"%(name))
            df.to_pickle(pklFileName)
        
        if save_csv:
            csvFileName = os.path.join(resultDir, "%s.csv"%(name))
            df.to_csv(csvFileName)
    
    #write scen error 
    if len(genScenErrDates):
        scenErrFile = os.path.join(resultDir, "scenErr.txt")
        with open(scenErrFile, 'wb') as fout:
            fout.write(scenErrStringIO.getvalue())
    scenErrStringIO.close()
    
    #generating summary files
    summary = {"n_rv": n_rv,
               "n_stock": n_stock,
               "T": T,
               "scenario": n_scenario,
               "alpha": alpha,
               "symbols":  ",".join(symbols),
               "transDates": [pd.to_datetime(t).strftime("%Y%m%d") 
                              for t in transDates],    #(T+1)
               "hist_period": hist_period,
               "buyTransFee":buyTransFee[0], 
               "sellTransFee":sellTransFee[0],
               "final_wealth": finalWealth,
               "scenFunc": scenFunc,
               "scen_err_cnt":len(genScenErrDates),
               "scen_err_dates": genScenErrDates,
               "machine": platform.node(),
               "elapsed": time.time()-t0
               }
    
    fileName = os.path.join(resultDir, 'summary.json')
    with open (fileName, 'w') as fout:
        json.dump(summary, fout, indent=4)
    
    print "%s-%s n%s-p%s-s%s-a%s --scenFunc %s --solver %s\nsimulation ok, %.3f secs"%(
             startDate, endDate, n_stock, hist_period, n_scenario, alpha,
             scenFunc, solver, time.time()-t0)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='dynamicSymbolSPPortfolio')
#     parser.parse_args()
    parser.add_argument('-n', '--symbols', type=int, default=5, help="num. of symbols")
    parser.add_argument('-p', '--histPeriod', type=int, default=40, help="historical period")
    parser.add_argument('-s', '--scenario', type=int, default=200, help="num. of scenario")
    parser.add_argument('-a', '--alpha', type=float, default=0.95, help="confidence level of CVaR")
    parser.add_argument('--solver', choices=["glpk", "cplex"], default="cplex", help="solver for SP")
    parser.add_argument('--scenFunc', choices=["Moment", "Copula"], default="Moment", help="function for generating scenario")
    parser.add_argument('-m', '--marketvalue', type=int, choices=[200501, 201312], default=201312, help="market value type")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-y', '--year', type=int, choices=range(2005, 2013+1), help="experiment in year")
    group.add_argument('-f', '--full', action='store_true', help="from 2005~2013")
    args = parser.parse_args()

    # 把參數 number 的值印出來
    print args
        
    #market value top 20 (2013/12/31)
    if args.marketvalue == 201312:
        symbols = ['2330', '2317', '6505', '2412', '2454',
                '2882', '1303', '1301', '1326', '2881'
                ]
        
    elif args.marketvalue == 200501:
        symbols = [
                '2330', '2412', '2882', '6505', '2317',
                '2303', '2002', '1303', '1326', '1301',
                
#                 '2881', '2886', '2409', '2891', '2357',
#                 '2382', '3045', '2883', '2454', '2880',
#                 '2892', '4904', '2887', '2353', '2324',
#                 '2801', '1402', '2311', '2475', '2888',
#                 '2408', '2308', '2301', '2352', '2603',
#                 '2884', '2890', '2609', '9904', '2610',
#                 '1216', '1101', '2325', '2344', '2323',
#                 '2371', '2204', '1605', '2615', '2201',
                ]
        
    if args.year:
        startDate = date(args.year, 1, 1)
        endDate = date(args.year, 12, 31)
    elif args.full:
        startDate = date(2005, 1, 1)
        endDate = date(2013, 12, 31)
        
    money = 1e6
    hist_period = args.histPeriod
    n_scenario = args.scenario
    buyTransFee=0.001425
    sellTransFee=0.004425
    alpha = args.alpha
    scenFunc = args.scenFunc
    solver = args.solver
    debug = False
    dynamicSymbolSPPortfolio(symbols, startDate, endDate,  money,
                           hist_period, n_scenario,
                           buyTransFee, sellTransFee,
                           alpha, scenFunc, solver, 
                           n_stock=args.symbols, 
                           save_pkl=False, save_csv=True, debug=debug)
        