# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
為了加速, 自行將SP轉成LP求解
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

PklBasicFeaturesDir = os.path.join(os.getcwd(),'pkl', 'BasicFeatures')


def MinCVaRPortfolioSP(symbols, riskyRet, riskFreeRet, allocatedWealth,
                       depositWealth, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet, n_scenario, 
                       solver="glpk"):
    '''
    two-stage stochastic programming
    
    variable: 
        N: num. of symbols,
        S: num. of scenarios
        
    symbols, list of string,
    riskRet, numpy.array, size: N 
    riskFreeRet, float
    allocatedWealth, numpy.array, size: N
    depositWealth, float,
    buyTransFee, numpy.array, size: N
    sellTransFee, numpy.array, size: N
    alpha, float,
    predictRiskRet, numpy.array, size: N * S
    predictRiskFreeRet, float 
    
    '''
    t = time.time()
    # Model
    model = ConcreteModel()
    
    #Set
    model.symbols = range(len(symbols))
    model.scenarios = range(n_scenario)
    
    #decision variables
    model.buys = Var(model.symbols, within=NonNegativeReals)        #stage 1
    model.sells = Var(model.symbols, within=NonNegativeReals)       #stage 1
    model.riskyWealth = Var(model.symbols, within=NonNegativeReals) #stage 2
    model.riskFreeWealth = Var(within=NonNegativeReals)             #stage 2
    
    #aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    model.Z = Var()
    
    #aux variable, portfolio wealth smaller (<=) than VaR
    model.Ys = Var(model.scenarios, within=NonNegativeReals)                 
    
    #constraint
    def riskyWeathConstraint_rule(model, n):
        '''
        riskyWealth is a decision variable depending on both buys and sells.
        (it means riskyWealth depending on scenario).
        therefore 
        buys and sells are fist stage variable,
        riskywealth is second stage variable
        '''
        return (model.riskyWealth[n] == 
                (1. + riskyRet[n]) * allocatedWealth[n] + 
                model.buys[n] - model.sells[n])
    
    model.riskyWeathConstraint = Constraint(model.symbols)
    
    
    def riskFreeWealthConstraint_rule(model):
        '''
        riskFreeWealth is decision variable depending on both buys and sells.
        therefore 
        buys and sells are fist stage variable,
        riskFreewealth is second stage variable
        '''
        totalSell = sum((1 - sellTransFee[n]) * model.sells[n] 
                        for n in model.symbols)
        totalBuy = sum((1 + buyTransFee[n]) * model.buys[n] 
                       for n in model.symbols)
            
        return (model.riskFreeWealth == 
                (1. + riskFreeRet)* depositWealth  + 
                totalSell - totalBuy)
            
    model.riskFreeWealthConstraint = Constraint()
    
    
    def CVaRConstraint_rule(model, s):
        '''auxiliary variable Y depends on scenario. CVaR <= VaR'''
        wealth = sum( (1. + predictRiskyRet[n][s] ) * model.riskyWealth[n] 
                     for n in model.symbols)
        return model.Ys[s] >= (model.Z - wealth)
    
    model.CVaRConstraint = Constraint(model.scenarios)
    
    #objective
    def TotalCostObjective_rule(model):
        return model.Z - 1/(1-alpha)* sum(model.Ys[s]/n_scenario for s in xrange(n_scenario))
        
    model.TotalCostObjective = Objective(sense=maximize)
    
    # Create a solver
    opt = SolverFactory(solver)
#     print "opt options:", opt.option
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
    
    display(instance)
    #output file to yaml format
#     results.write()
    print "MinCVaRPortfolioSP elapsed %.3f secs"%(time.time()-t)
 

def constructModelData():
    symbols = ('1101', "1102")
    N = len(symbols)
    allocated = np.zeros(N)
    money = 1e6
    
    buyTransFee =  np.ones(N)*0.003  #買進0.1425%手續費
    sellTransFee =  np.ones(N)*0.004425  #賣出0.3%手續費+0.1425%交易稅
    
    riskyRet = [0.1, 0.2]
    riskFreeRet = 0
    alpha = 0.95 
    predictRiskyRet = np.array([[0.1422, 0.0389], 
                       [0.2582, 0.0266],
                       [0.01292, 0.0347],
                       [0.0381, 0.0643],
                       [0.0473, 0.1013],
                       ]).T
#     print predictRiskyRet
    predictRiskFreeRet = 0.
    
    
    MinCVaRPortfolioSP(symbols, riskyRet, riskFreeRet, allocated,
                       money, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet, 
                       n_scenario = 5,
                       solver="cplex")
 
if __name__ == '__main__':
    constructModelData()