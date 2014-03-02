# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
minimize CVaR risk


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
model.sells = Var(model.symbols)       #stage 1
model.riskyWealth = Var(model.symbols, within=NonNegativeReals) #stage 2
model.riskFreeWealth = Var(within=NonNegativeReals)             #stage 2
model.FirstStageCost = Var()
model.SecondStageCost = Var()

def shortsellConstriant_rule(model, m):
    return max(models.sells[m], 0)

model.shortsellConstriant = Constraint(model.symbols)


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
        
model.riskyWeathConstraint = Constraint(model.symbols)
model.riskFreeWealthConstraint = Constraint()

# Stage-specific 
def ComputeFirstStageWealth_rule(model):
    return model.FirstStageCost  == 0.0

def ComputeSecondStageWealth_rule(model):
    '''total wealth at the beginning of time (t+1) '''
    wealth1 = sum( (1. + model.predictRiskyRet[m] ) * model.riskyWealth[m] 
                 for m in model.symbols)
    wealth2 = (1.+ model.predictRiskFreeRet ) * model.riskFreeWealth
    return model.SecondStageCost - wealth1 - wealth2 == 0

model.ComputeFirstStageWealth = Constraint()
model.ComputeSecondStageWealth = Constraint()

#objective
def TotalWealthObjective_rule(model):
    return model.FirstStageCost + model.SecondStageCost
    
model.TotalWealthObjective = Objective(sense=maximize)
