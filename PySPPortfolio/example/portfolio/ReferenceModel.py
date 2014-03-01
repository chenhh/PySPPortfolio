# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''

from coopr.pyomo import *

# Model
model = AbstractModel()

# Parameters
model.symbols = Set()

#return of current period, known
model.riskyRet = Param(model.symbols)

#return of next period, uncertain
model.tp1RiskyRet = Param(models.symbols) 
model.allocatedWealth = Param(model.symbols)
model.depositWealth = Param()
model.riskFreeRet = Param()
model.buyTransFee = Param(model.symbols)
model.sellTransFee = Param(model.symbols)

#decision variables
model.buys = Var(model.symbols, within=NonNegativeReals)
model.sells = Var(model.symbols, within=NonNegativeReals)
model.riskyWealth = Var(model.symbols, within=NonNegativeReals)
model.riskFreeWealth = Var(within=NonNegativeReals)
model.FirstStageWealth = Var()
model.SecondStageWealth = Var()

#constraint
def riskyWeathConstraint_rule(model, m):
    '''
    riskyWealth is decision variable depending on both buys and sells.
    therefore 
    buys and sells are fist stage variable,
    riskywealth is second stage variable
    '''
    return (model.riskyWealth[m] == 
            (1. + model.riskyRet[m]) * model.allocatedWealth[m] + 
            model.buys[m] - model.sells[m])
    
def riskFreeWealthConstraint_rule(model):
    '''
    riskFreeWealth is decision variable depending on both buys and sells.
    therefore 
    buys and sells are fist stage variable,
    riskFreewealth is second stage variable
    '''
    totalSell = sum((1.-model.sellTransFee[m])*model.sells[m] 
                    for m in model.symbols)
    totalBuy = sum((1+model.buyTransFee[m])*model.buys[m] 
                   for m in model.symbols)
        
    return (model.riskFreeWealth == 
            (1. + model.riskFreeRet)* model.depositWealth  + 
            totalSell - totalBuy)
        
model.riskyWeathConstraint = Constraint(model.symbols)
model.riskFreeWealthConstraint = Constraint()

# Stage-specific 
def ComputeFirstStageWealth_rule(model):
    '''
    '''
    return (model.FirstStageWealth - sum(model.riskyWealth) - 
            model.riskFreeWealth) == 0.0 

def ComputeSecondStageWealth_rule(model):
    '''
    '''
    wealth = sum( (1. + model.riskyRet[m]) * model.riskyWealth[m] 
                 for m in model.symbols)
    wealth += (1.+ model.riskFreeRet) * model.riskFreeWealth
    return model.SecondStageWealth - wealth == 0

model.ComputeFirstStageWealth = Constraint()
model.ComputeSecondStageWealth = Constraint()

#objective
def TotalWealthObjective_rule(model):
    return model.FirstStageWealth + model.SecondStageWealth
    
model.TotalWealthObjective = Objective(sense=maximize)
