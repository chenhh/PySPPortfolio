# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
worst case CVaR
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



def WorstCVaRPortfolioSP(symbols, riskyRet, riskFreeRet, allocatedWealth,
                       depositWealth, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet, n_scenario, 
                       probs=None, solver="cplex"):
    '''
    two-stage stochastic programming
    
    variable: 
        M: num. of symbols,
        S: num. of scenarios
        
    symbols, list of string,
    riskRet, numpy.array, size: M
    riskFreeRet, float
    allocatedWealth, numpy.array, size: M
    depositWealth, float,
    buyTransFee, numpy.array, size: M
    sellTransFee, numpy.array, size: M
    alpha, float,
    predictRiskRet, numpy.array, size: L * M * S, L個不確定的dist.
    predictRiskFreeRet, float 
    probs, numpy.array, size: S
    solver, string in {glpk or cplex}
    '''
    t = time.time()
    assert len(predictRiskyRet.shape) == 3
    n_dist = predictRiskyRet.shape[0] 
    
    if not probs:
        probs = np.ones(n_scenario, dtype=np.float)/n_scenario
    
    # Model
    model = ConcreteModel()
    
    #Set
    model.symbols = range(len(symbols))
    model.scenarios = range(n_scenario)
    model.distributions = range(n_dist)
    
    #decision variables
    model.buys = Var(model.symbols, within=NonNegativeReals)        #stage 1
    model.sells = Var(model.symbols, within=NonNegativeReals)       #stage 1
    model.riskyWealth = Var(model.symbols, within=NonNegativeReals) #stage 2
    model.riskFreeWealth = Var(within=NonNegativeReals)             #stage 2
    
    #aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    model.Z = Var()
    
    #aux variable, portfolio wealth less than than VaR (Z)
    model.Ys = Var(model.distributions, model.scenarios, within=NonNegativeReals)                 
    
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
    
    
    def WCVaRConstraint_rule(model, d, s):
        '''auxiliary variable Y depends on scenario. CVaR <= VaR'''
        wealth = sum( (1. + predictRiskyRet[d, m, s] ) * model.riskyWealth[m] 
                     for m in model.symbols)
        return model.Ys[d, s] >= (model.Z - wealth)
    
    model.WCVaRConstraint = Constraint(model.distributions, model.scenarios)
    
    #objective
    def WCVaRObjective_rule(model):
        val = 0
        for d in model.distributions:
            for s in model.scenarios:
                val += model.Ys[d, s] * probs[s]
        val *= 1/(1-alpha)
        val = model.Z - val
        return val
     
    model.WCVaRObjective = Objective(sense=maximize)
    
    # Create a solver
    opt = SolverFactory(solver)
    
    if solver =="cplex":
        opt.options["threads"] = 4
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
    WCVaR = results.Solution.Objective.__default_objective__['value']
#     display(instance)
    M = len(symbols)
    results = {"WCVaR": WCVaR}
    
    for v in instance.active_components(Var):
#         print "Variable",v
        varobject = getattr(instance, v)
        if v == "buys":
            results[v] = np.fromiter((varobject[index].value for index in varobject), np.float)
        elif v == "sells":
            results[v] = np.fromiter((varobject[index].value for index in varobject), np.float)
        elif v == "Z":
            results["VaR"] = varobject.value
#     print results
    
    print "WCVaR:", WCVaR 
    print "WorstCVaRPortfolioSP elapsed %.3f secs"%(time.time()-t)
    return  results


def testWCVaR():
    symbols = ('1101', "1102", '1103')
    M = len(symbols)
    allocated = np.zeros(M)
    money = 1e6
    
    buyTransFee =  np.ones(M)*0.001425  #買進0.1425%手續費
    sellTransFee =  np.ones(M)*0.004425  #賣出0.3%手續費+0.1425%交易稅
    
    riskyRet = [0.01, 0.02, 0.015]
    riskFreeRet = 0
    alpha = 0.95 
    predictRiskyRet1 = np.array(
                        [
                       [0.1422, 0.0389, 0.0323], 
                       [0.2582, 0.0266, -0.01234],
                       [0.01292, 0.0347, 0.0013],
                       [0.0381, 0.0643, -0.0023],
                       [0.0473, 0.1013, 0.0012],
                       ]).T
    predictRiskyRet2 = np.array(
                        [
                       [0.1422, 0.0389, 0.0323], 
                       [0.2582, 0.0266, -0.01234],
                       [0.01292, 0.0347, 0.0013],
                       [0.0381, 0.0643, -0.0023],
                       [0.0473, 0.1013, 0.0012],
                       ]).T                   
    predictRiskyRet = np.array((predictRiskyRet1, predictRiskyRet2))
    
#     print predictRiskyRet
    predictRiskFreeRet = 0.
    
    
    results = WorstCVaRPortfolioSP(symbols, riskyRet, riskFreeRet, allocated,
                       money, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet, 
                       n_scenario = 5,
                       solver="cplex")
    print results
    print "*"*80

if __name__ == '__main__':
    testWCVaR()