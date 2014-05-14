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


def MinCVaRPortfolioSIP(symbols, riskyRet, riskFreeRet, allocatedWealth,
                       depositWealth, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet, n_scenario, 
                       probs=None, solver="glpk", n_stock=5):
    '''
    two-stage stochastic integer programming
    given M stocks, choose the best n_stocks to invest
    
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
    predictRiskRet, numpy.array, size: M * S
    predictRiskFreeRet, float 
    n_scenario, positive integer
    probs, numpy.array, size: S
    solver, string in {glpk or cplex}
    n_stock, positive integer, max number of stock to invest in
    
    '''
    t = time.time()
    if not probs:
        probs = np.ones(n_scenario, dtype=np.float)/n_scenario
    
    # Model
    model = ConcreteModel()
    
    #Set
    model.symbols = range(len(symbols))
    model.scenarios = range(n_scenario)
    
    #decision variables
    #stage1
    #0: stock is not chosen, 1: stock is chosen
    model.chosen = Var(model.symbols, within=Binary)   
    model.buys = Var(model.symbols, within=NonNegativeReals)        
    model.sells = Var(model.symbols, within=NonNegativeReals)
    
    #stage 2
    model.riskyWealth = Var(model.symbols, within=NonNegativeReals) 
    model.riskFreeWealth = Var(within=NonNegativeReals)
    
    #aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    model.Z = Var()
    
    #aux variable, portfolio wealth less than than VaR (Z)
    model.Ys = Var(model.scenarios, within=NonNegativeReals)                 
    
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
    
    
    def CVaRConstraint_rule(model, s):
        '''auxiliary variable Y depends on scenario. CVaR <= VaR'''
        wealth = sum( (1. + predictRiskyRet[m][s] ) * model.riskyWealth[m] 
                     for m in model.symbols)
        return model.Ys[s] >= (model.Z - wealth)
    
    model.CVaRConstraint = Constraint(model.scenarios)
    
    
    def chosenConstraint_rule(model, m):
        totalCapital = sum(allocatedWealth) + depositWealth
        return (model.riskyWealth[m] <= model.chosen[m] * totalCapital)  
    
    model.chosenConstraint = Constraint(model.symbols)
    
    def portoflioSizeConstraint_rule(model):
        '''restricted the number of stock to invest'''
        return sum(model.chosen[m] for m in model.symbols) <= n_stock
    
    model.portoflioSizeConstraint = Constraint()
    
    #objective
    def TotalCostObjective_rule(model):
        return model.Z - 1/(1-alpha)* sum(model.Ys[s] * probs[s] 
                                          for s in xrange(n_scenario))
        
    model.TotalCostObjective = Objective(sense=maximize)
    
    # Create a solver
    opt = SolverFactory(solver)
    if solver =="cplex":
        opt.options["threads"] = 6
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
    CVaR = results.Solution.Objective.__default_objective__['value']
#     display(instance)
    M = len(symbols)
    results = {"CVaR": CVaR}
    
    for v in instance.active_components(Var):
        varobject = getattr(instance, v)
        if v == "buys":
            results[v] = np.fromiter((varobject[index].value for index in varobject), np.float)
        elif v == "sells":
            results[v] = np.fromiter((varobject[index].value for index in varobject), np.float)
        elif v == "Z":
            results["VaR"] = varobject.value
    
    print "CVaR:", CVaR 
    print "MinCVaRPortfolioSIP elapsed %.3f secs"%(time.time()-t)
    return results
 

def MinCVaRPortfolioSP(symbols, riskyRet, riskFreeRet, allocatedWealth,
                       depositWealth, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet, n_scenario, 
                       probs=None, solver="glpk"):
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
    predictRiskRet, numpy.array, size: M * S
    predictRiskFreeRet, float 
    probs, numpy.array, size: S
    solver, string in {glpk or cplex}
    '''
    t = time.time()
    if not probs:
        probs = np.ones(n_scenario, dtype=np.float)/n_scenario
    
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
    
    #aux variable, portfolio wealth less than than VaR (Z)
    model.Ys = Var(model.scenarios, within=NonNegativeReals)                 
    
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
    
    
    def CVaRConstraint_rule(model, s):
        '''auxiliary variable Y depends on scenario. CVaR <= VaR'''
        wealth = sum( (1. + predictRiskyRet[m][s] ) * model.riskyWealth[m] 
                     for m in model.symbols)
        return model.Ys[s] >= (model.Z - wealth)
    
    model.CVaRConstraint = Constraint(model.scenarios)
    
    #objective
    def CVaRObjective_rule(model):
        return model.Z - 1/(1-alpha)* sum(model.Ys[s] * probs[s] 
                                          for s in xrange(n_scenario))
        
    model.CVaRObjective = Objective(sense=maximize)
    
    # Create a solver
    opt = SolverFactory(solver)
    
    if solver =="cplex":
        opt.options["threads"] = 6
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
    CVaR = results.Solution.Objective.__default_objective__['value']
#     display(instance)
    M = len(symbols)
    results = {"CVaR": CVaR}
    
    for v in instance.active_components(Var):

        varobject = getattr(instance, v)
        if v == "buys":
            results[v] = np.fromiter((varobject[index].value for index in varobject), np.float)
        elif v == "sells":
            results[v] = np.fromiter((varobject[index].value for index in varobject), np.float)
        elif v == "Z":
            results["VaR"] = varobject.value

    
    print "CVaR:", CVaR 
    print "MinCVaRPortfolioSP elapsed %.3f secs"%(time.time()-t)
    return  results


def MinCVaRPortfolioSP2(symbols, riskyRet, riskFreeRet, allocatedWealth,
                       depositWealth, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet, n_scenario, 
                       probs=None, solver="glpk"):
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
    predictRiskRet, numpy.array, size: M * S
    predictRiskFreeRet, float 
    probs, numpy.array, size: S
    solver, string in {glpk or cplex}
    '''
    t = time.time()
    if not probs:
        probs = np.ones(n_scenario, dtype=np.float)/n_scenario
    
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
    
    #aux variable, portfolio wealth less than than VaR (Z)
    model.Ys = Var(model.scenarios, within=NonNegativeReals)                 
    
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
    
    
    def CVaRConstraint_rule(model, s):
        '''auxiliary variable Y depends on scenario. CVaR <= VaR'''
        wealth = sum( (1. + predictRiskyRet[m][s] ) * model.riskyWealth[m] 
                     for m in model.symbols)
        return model.Ys[s] >= (-wealth - model.Z )
    
    model.CVaRConstraint = Constraint(model.scenarios)
    
    #objective
    def CVaRObjective_rule(model):
        return model.Z + 1/(1-alpha)* sum(model.Ys[s] * probs[s] 
                                          for s in xrange(n_scenario))
        
    model.CVaRObjective = Objective(sense=minimize)
    
    # Create a solver
    opt = SolverFactory(solver)
    
    if solver =="cplex":
        opt.options["threads"] = 6
    
    instance = model.create()
    results = opt.solve(instance)  
    instance.load(results)
    CVaR = results.Solution.Objective.__default_objective__['value']
#     display(instance)
    M = len(symbols)
    results = {"CVaR": CVaR}
    
    for v in instance.active_components(Var):

        varobject = getattr(instance, v)
        if v == "buys":
            results[v] = np.fromiter((varobject[index].value for index in varobject), np.float)
        elif v == "sells":
            results[v] = np.fromiter((varobject[index].value for index in varobject), np.float)
        elif v == "Z":
            results["VaR"] = varobject.value

    
    print "CVaR:", CVaR 
    print "MinCVaRPortfolioSP elapsed %.3f secs"%(time.time()-t)
    return  results


def testCVaR():
    symbols = ('1101', "1102", '1103')
    M = len(symbols)
    allocated = np.zeros(M)
    money = 1e6
    
    buyTransFee =  np.ones(M)*0.001425  #買進0.1425%手續費
    sellTransFee =  np.ones(M)*0.004425  #賣出0.3%手續費+0.1425%交易稅
    
    riskyRet = [0.01, 0.02, 0.015]
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
    
    
    results = MinCVaRPortfolioSP(symbols, riskyRet, riskFreeRet, allocated,
                       money, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet, 
                       n_scenario = 5,
                       solver="cplex")
    print results
    print "*"*80
     
    results2 = MinCVaRPortfolioSP2(symbols, riskyRet, riskFreeRet, allocated,
                       money, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet, 
                       n_scenario = 5,
                       solver="cplex")
    print results2
 
if __name__ == '__main__':
    testCVaR()