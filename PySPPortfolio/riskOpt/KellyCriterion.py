# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw

kelly multivariate investment
'''

from __future__ import division
from coopr.pyomo import *
from time import time
from datetime import date
import numpy as np
import pandas as pd
import os
import time
from coopr.opt import  SolverFactory


def KellyCriterion(symbols, riskyRet, money=1e6, solver="cplex"):
    '''
    @riskyRet, shape: M*T
    
    maximize W*R - 1/2W^T \simga W
    '''
    t = time.time()
    
    
    
    mu = riskyRet.mean(axis=1)
    print "mu:", mu
    model = ConcreteModel()
    
    #Set
    model.symbols = range(len(symbols))
       
    #decision variables
    model.W = Var(model.symbols, within=NonNegativeReals)
    
    #constraint
    def CapitalConstraint_rule(model):
        allocation = sum(model.W[idx] for idx in model.symbols)
        return allocation == money
    
    model.CapitalConstraint = Constraint()
    
    
    #objective
    def KellyObjective_rule(model):
        profit = sum(model.W[idx]*mu[idx] for idx in model.symbols)
        risk = 0
        for idx in model.symbols:
            for jdx in model.symbols:
                    risk += model.W[idx] * model.W[jdx] * mu[idx]* mu[jdx] 
        
        return profit - 1./2 * risk
        
    model.KellyObjective = Objective(sense=maximize)
    
    # Create a solver
    opt = SolverFactory(solver)
    
    if solver =="cplex":
        opt.options["threads"] = 4
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
    obj = results.Solution.Objective.__default_objective__['value']
    display(instance)
    
    print "Kelly elapsed %.3f secs"%(time.time()-t)


def KellySP(symbols, riskyRet, riskFreeRet, allocatedWealth,
                    depositWealth, buyTransFee, sellTransFee,
                    predictRiskyRet, predictRiskFreeRet, n_scenario, 
                   probs=None, solver="cplex"):
    '''
    M: n_symbol, S: n_scenario
    
    @symbols, list, size: M
    @riskRet: np.array, size: M
    @riskFreeRet: float
    @allocatedWealth, np.array, size:M
    @depositWealth, float
    @buyTransFee, np.array, size:M
    @sellTransFee, np.array, size:M
    @riskyRet, shape: M*T
    @predictRiskyRet, np.array, size: M*S
    @predictRiskFreeRet, float
    @n_scneario, int
    @probs: np.array, size: S
    maximize E(W*R - 1/2W^T \simga W)
    '''
    t = time.time()
    
    if not probs:
        probs = np.ones(n_scenario, dtype=np.float)/n_scenario
    

    model = ConcreteModel()
    
    #Set
    model.symbols = range(len(symbols))
       
    model.scenarios = range(n_scenario)
    
    #decision variables
    #stage 1  
    model.buys = Var(model.symbols, within=NonNegativeReals)        
    model.sells = Var(model.symbols, within=NonNegativeReals)
    
    #stage 2
    model.riskyWealth = Var(model.symbols, within=NonNegativeReals) 
    model.riskFreeWealth = Var(within=NonNegativeReals)
    
    #constraint
    def riskyWeathConstraint_rule(model, m):
        '''
        riskyWealth is a decision variable depending on both buys and sells.
        (it means riskyWealth depending on scenario).
        therefore 
        buys and sells are fist stage variable,
        riskywealth is second stage variable
        '''
        return (model.riskyWealth[m] == 
                (1. + riskyRet[m]) * allocatedWealth[m] + 
                model.buys[m] - model.sells[m])
    
    model.riskyWeathConstraint = Constraint(model.symbols)
    
    
    def riskFreeWealthConstraint_rule(model):
        '''
        riskFreeWealth is decision variable depending on both buys and sells.
        therefore 
        buys and sells are fist stage variable,
        riskFreewealth is second stage variable
        '''
        totalSell = sum((1 - sellTransFee[m]) * model.sells[m] 
                        for m in model.symbols)
        totalBuy = sum((1 + buyTransFee[m]) * model.buys[m] 
                       for m in model.symbols)
            
        return (model.riskFreeWealth == 
                (1. + riskFreeRet)* depositWealth  + 
                totalSell - totalBuy)
            
    model.riskFreeWealthConstraint = Constraint()
    
    #objective
    def TotalCostObjective_rule(model):
        '''' E(W*R - 1/2W^T \simga W) '''
        profit = sum(probs[s]* model.riskyWealth[symbol]* predictRiskyRet[symbol, s]
                  for symbol in model.symbols
                  for s in xrange(n_scenario))
        
        risk = 0
        for idx in model.symbols:
            for jdx in model.symbols:
                for s in xrange(n_scenario):
                    risk += (model.riskyWealth[idx] * model.riskyWealth[jdx] *
                              predictRiskyRet[idx, s]*predictRiskyRet[jdx, s])
        return profit - 1./2*risk
    
    model.TotalCostObjective = Objective(sense=maximize)
    
    # Create a solver
    opt = SolverFactory(solver)
    
    if solver =="cplex":
        opt.options["threads"] = 4
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
    obj = results.Solution.Objective.__default_objective__['value']
    display(instance)
    
    print "KellySP elapsed %.3f secs"%(time.time()-t)
    

def testKelly():
    FileDir = os.path.abspath(os.path.curdir)
    PklBasicFeaturesDir = os.path.join(FileDir, '..', 'pkl', 'BasicFeatures')
    
    symbols = ['2330', '2317', '6505']
#     symbols = [
#                 '2330', '2412', '2882', '6505', '2317',
#                 '2303', '2002', '1303', '1326', '1301',
#                 '2881', '2886', '2409', '2891', '2357',
#                 '2382', '3045', '2883', '2454', '2880',
#                 '2892', '4904', '2887', '2353', '2324',
#                 '2801', '1402', '2311', '2475', '2888',
#                 '2408', '2308', '2301', '2352', '2603',
#                 '2884', '2890', '2609', '9904', '2610',
#                 '1216', '1101', '2325', '2344', '2323',
#                 '2371', '2204', '1605', '2615', '2201',
#         ]
    n_period = 2000
    ROIs = np.empty((len(symbols)+1, n_period))
    for idx, symbol in enumerate(symbols):
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, '%s.pkl'%symbol))
        roi =  df['adjROI'][:n_period]
        ROIs[idx] = roi
    
    ROIs[-1] = np.zeros(n_period)
    symbols.append('deposit')
    
    KellyCriterion(symbols, ROIs, money=1e6, solver="cplex")


def testKellySP():
    symbols = ('1101', "1102", '1103')
    M = len(symbols)
    allocated = np.zeros(M)
    money = 1e6
    
    buyTransFee =  np.ones(M)*0.001425  #買進0.1425%手續費
    sellTransFee =  np.ones(M)*0.004425  #賣出0.3%手續費+0.1425%交易稅
    
    riskyRet = np.array([0.01, 0.02, 0.015])
    riskFreeRet = 0
    alpha = 0.95 
    predictRiskyRet = np.array([[0.1422, 0.0389, 0.0323], 
                       [0.2582, 0.0266, -0.01234],
                       [0.01292, 0.0347, 0.0013],
                       [0.0381, 0.0643, -0.0023],
                       [0.0473, 0.1013, 0.0012],
                       ]).T
#     print predictRiskyRet
    predictRiskFreeRet = 0.
    
    
    results = KellySP(symbols, riskyRet, riskFreeRet, allocated,
                       money, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet, 
                       n_scenario = 5,
                       solver="cplex")
    print results
    print "*"*80


if __name__ == '__main__':
#     testKelly()
    testKellySP()