# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
為了加速, 自行將SP轉成LP求解
'''

from coopr.pyomo import *
from time import time

def MinCVaRPortfolioSP(symbols, riskyRet, riskFreeRet, allocatedWealth,
                       depositWealth, buyTransFee, sellTransFee, alpha,
                       predictRiskyRet, predictRiskFreeRet):
    '''
    variable: 
        n: num. of symbols, 
        T: num. of historical periods
        S: num. of scenarios
        
    symbols, list of string,
    riskRet, numpy.array, size: n * T
    riskFreeRet, numpy.array, size: T
    allocatedWealth, numpy.array, size: n
    depositWealth, float,
    buyTransFee, float,
    sellTransFee, float
    alpha, float
    predictRiskRet, numpy.array, size: n*S
    predictRiskFreeRet, float 
    
    '''
    t = time()
    # Model
    model = ConcreteModel()
    
    #Set
    model.symbols = range(len(symbols))
    
    #decision variables
    model.buys = Var(model.symbols, within=NonNegativeReals)        #stage 1
    model.sells = Var(model.symbols, within=NonNegativeReals)       #stage 1
    model.riskyWealth = Var(model.symbols, within=NonNegativeReals) #stage 2
    model.riskFreeWealth = Var(within=NonNegativeReals)             #stage 2
    
    #aux variable, variable in definition of CVaR, equals to VaR at opt. sol.
    model.Z = Var()
    
    #aux variable, portfolio wealth smaller (<=) than VaR
    model.Ys = Var(within=NonNegativeReals)                 
    
    
    #constraint
    def riskyWeathConstraint_rule(model, m):
        '''
        riskyWealth is a decision variable depending on both buys and sells.
        (it means riskyWealth depending on scenario).
        therefore 
        buys and sells are fist stage variable,
        riskywealth is second stage variable
        '''
        return (riskyWealth[m] == 
                (1. + model.riskyRet[m]) * allocatedWealth[m] + 
                model.buys[m] - model.sells[m])
    
    model.riskyWeathConstraint = Constraint(model.symbols)
    
    
    def riskFreeWealthConstraint_rule(model):
        '''
        riskFreeWealth is decision variable depending on both buys and sells.
        therefore 
        buys and sells are fist stage variable,
        riskFreewealth is second stage variable
        '''
        totalSell = sum((1. - sellTransFee[m]) * model.sells[m] 
                        for m in model.symbols)
        totalBuy = sum((1 + buyTransFee[m]) * model.buys[m] 
                       for m in model.symbols)
            
        return (model.riskFreeWealth == 
                (1. + riskFreeRet)* depositWealth  + 
                totalSell - totalBuy)
            
    model.riskFreeWealthConstraint = Constraint()
    
    
    def CVaRConstraint_rule(model):
        '''auxiliary variable Y depends on scenario. CVaR <= VaR
        '''
        wealth = sum( (1. + predictRiskyRet[m] ) * model.riskyWealth[m] 
                     for m in model.symbols)
        return model.Ys >= (model.Z - wealth)
    
    model.CVaRConstraint = Constraint()
    
    #objective
    def TotalCostObjective_rule(model):
        return model.Z +-1/(1-alpha)* model.Ys
        
    model.TotalCostObjective = Objective(sense=maximize)
    
    
    # Create a solver
    opt = SolverFactory('glpk')
    print "opt options:", opt.option
    
    instance = model.create()
    results = opt.solve(instance)
#     print type(instance)
#     print dir(instance)
#     print results
#     print 
#     print results.Solution.Objective.x1.Value
#     print results.Solver.Status
#     print results.Solution.Status
#     print type(results)
#     print dir(results)
    
    instance.load(results)
    
    display(instance)
    #output file to yaml format
#     results.write()
    print "MinCVaRPortfolioSP elapsed %.3f secs"%(time.time()-t)
 