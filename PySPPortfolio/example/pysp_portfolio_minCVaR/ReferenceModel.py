# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
minimize the CVaR

'''

from coopr.pyomo import *

# Model
model = AbstractModel()
model.symbols = Set()

# Parameters
#return of current period, known
model.riskyRet = Param(model.symbols)
model.riskFreeRet = Param()

#return of next period, uncertain
model.predictRiskyRet = Param(model.symbols) 
model.predictRiskFreeRet = Param()

model.allocatedWealth = Param(model.symbols)
model.depositWealth = Param()

model.buyTransFee = Param(model.symbols)
model.sellTransFee = Param(model.symbols)

#decision variables
model.buys = Var(model.symbols, within=NonNegativeReals)        #stage 1
model.sells = Var(model.symbols, within=NonNegativeReals)       #stage 1
model.riskyWealth = Var(model.symbols, within=NonNegativeReals) #stage 2
model.riskFreeWealth = Var(within=NonNegativeReals)             #stage 2
model.Z = Var() #aux variable
model.Ys = Var(within=NonNegativeReals)
model.FirstStageCost = Var()
model.SecondStageCost = Var()

#constraint
def riskyWeathConstraint_rule(model, m):
    '''
    riskyWealth is a decision variable depending on both buys and sells.
    therefore 
    buys and sells are fist stage variable,
    riskywealth is second stage variable
    '''
    return (model.riskyWealth[m] == 
            (1. + model.riskyRet[m]) * model.allocatedWealth[m] + 
            model.buys[m] - model.sells[m])

model.riskyWeathConstraint = Constraint(model.symbols)

def riskFreeWealthConstraint_rule(model):
    '''
    riskFreeWealth is decision variable depending on both buys and sells.
    therefore 
    buys and sells are fist stage variable,
    riskFreewealth is second stage variable
    '''
    totalSell = sum((1 - model.sellTransFee[m]) * model.sells[m] 
                    for m in model.symbols)
    totalBuy = sum((1 + model.buyTransFee[m]) * model.buys[m] 
                   for m in model.symbols)
        
    return (model.riskFreeWealth == 
            (1. + model.riskFreeRet)* model.depositWealth  + 
            totalSell - totalBuy)
        
model.riskFreeWealthConstraint = Constraint()

def CVaRConstraint_rule(model):
    wealth = sum( (1. + model.predictRiskyRet[m] ) * model.riskyWealth[m] 
                 for m in model.symbols)
    return model.Ys >= wealth

model.CVaRConstraint = Constraint()

def CVaRConstraint1_rule(model):

# Stage-specific 
def ComputeFirstStageCost_rule(model):
    return model.FirstStageCost  == 0.0

model.ComputeFirstStageCost = Constraint()

def ComputeSecondStageCost_rule(model):
    '''CVaR time (t+1) '''
    
    return model.SecondStageCost  == model.Z - 1/(1-.95)* model.Ys

  model.ComputeSecondStageCost = Constraint()

#objective
def TotalCostObjective_rule(model):
    return model.FirstStageCost + model.SecondStageCost
    
model.TotalCostObjective = Objective(sense=maximize)
