# -*- coding: utf-8 -*-
'''
@author: Hung-Hsin Chen
@mail: chenhh@par.cse.nsysu.edu.tw
'''
import time
import os
from datetime import date
import numpy as np
import pandas as pd
from coopr.pyomo import (Set, RangeSet, Param, Var, Objective, Constraint,
                         ConcreteModel, Reals, NonNegativeReals, maximize, display)
from coopr.opt import  SolverFactory

PklBasicFeaturesDir = os.path.join(os.getcwd(),'pkl', 'BasicFeatures')


def optimalMultiStagePortfolio(riskyRetMtx, riskFreeRetVec, 
                               buyTransFeeMtx, sellTransFeeMtx,
                               allocatedWealthVec):
    '''
    -假設資料共T期, 投資在M個risky assets, 以及1個riskfree asset
    -求出每個risky asset每一期應該要買進以及賣出的金額
    -最後一期結算不投資
    
    @param riskyRetMtx, numpy.array, size: M * T+1
    @param riskFreeRetVec, numpy.array, size: T+1
    @param buyTransFeeMtx, numpy.array, size: M * T
    @param sellTransFeeMtx, numpy.array, size: M * T
    @param allocatedWealthVec, numpy.array, size: (M+1) (最後一個為cash)
    
    @return (buyMtx, sellMtx), numpy.array, each size: M*T
    '''
    assert buyTransFeeMtx.shape == sellTransFeeMtx.shape
    assert riskyRetMtx.shape[1] == riskFreeRetVec.size
   
    M, T =  buyTransFeeMtx.shape

    t1 = time.time()
    #create model
    model = ConcreteModel()
    
    #number of asset and number of periods
    model.M = range(M)
    model.T = range(T)
    model.Tp1 = range(T+1)
    model.Mp1 = range(M+1)
    
    #decision variables
    model.buys = Var(model.M, model.T, within=NonNegativeReals)
    model.sells = Var(model.M, model.T, within=NonNegativeReals)
    model.riskyWealth = Var(model.M, model.T, within=NonNegativeReals)
    model.riskFreeWealth = Var(model.T, within=NonNegativeReals)
    
    #objective
    def objective_rule(model):
        wealth =sum( (1. + riskyRetMtx[m, T]) * model.riskyWealth[m, T-1] 
                     for m in xrange(M))
        wealth += (1.+ riskFreeRetVec[T]) * model.riskFreeWealth[T-1] 
        return wealth
        
    model.objective = Objective(rule=objective_rule, sense=maximize)
    
    #constraint
    def riskyWealth_constraint_rule(model, m, t):
        if t>=1:
            preWealth = model.riskyWealth[m, t-1]
        else:
            preWealth = allocatedWealthVec[m]
       
        return (model.riskyWealth[m, t] == 
                (1. + riskyRetMtx[m,t])*preWealth + 
                model.buys[m,t] - model.sells[m,t])
    
    def riskyFreeWealth_constraint_rule(model, t):
        totalSell = sum((1-sellTransFeeMtx[mdx, t])*model.sells[mdx, t] 
                        for mdx in xrange(M))
        totalBuy = sum((1+buyTransFeeMtx[mdx, t])*model.buys[mdx, t] 
                       for mdx in xrange(M))
        
        if t >=1:
            preWealth = model.riskFreeWealth[t-1]  
        else:
            preWealth = allocatedWealthVec[-1]
    
        return( model.riskFreeWealth[t] == 
                (1. + riskFreeRetVec[t])*preWealth + 
                totalSell - totalBuy)
        
    
    model.riskyWeathConstraint = Constraint(model.M, model.T, 
                                            rule=riskyWealth_constraint_rule)
    model.riskFreeWealthConstraint = Constraint(model.T, 
                                          rule=riskyFreeWealth_constraint_rule)
    
    #optimizer
    opt = SolverFactory('cplex')
#     opt.options["threads"] = 4
    
    instance = model.create()
    results = opt.solve(instance)
    print results
    instance.load(results)
    display(instance)
#     instance.load(results)
#     for var in instance.active_components(Var):
#         varobj = getattr(instance, var)
#         for idx in varobj:
#             print varobj[idx], varobj[idx].value
#    
    print "elapsed %.3f secs"%(time.time()-t1)


def constructModelData():
    symbols = ('1101', '1102', '1103')
    startDate, endDate  = date(2012,1,1), date(2012, 12, 31)
    
    dfs = []
    for symbol in symbols:
        df = pd.read_pickle(os.path.join(PklBasicFeaturesDir, 
                                    'BasicFeatures_%s_00-12.pkl'%symbol))
        dfs.append(df[startDate: endDate])
    
    M, T = len(symbols), dfs[0].index.size - 1
    money = 1e5
    
    riskyRetMtx = np.empty((M, T+1))
    for idx, df in enumerate(dfs):
        riskyRetMtx[idx, :] = df['adjROI'].values/100
    
    riskFreeRetVec = np.zeros(T+1) 
    buyTransFeeMtx = np.ones((M,T))* 0.003  #買進0.3%手續費
    sellTransFeeMtx = np.ones((M,T))* 0.004425  #賣出0.3%手續費+0.1425%交易稅
    allocated = np.zeros(M+1)
    allocated[-1] = money

    optimalMultiStagePortfolio(riskyRetMtx, riskFreeRetVec, 
                               buyTransFeeMtx, sellTransFeeMtx, allocated)
if __name__ == '__main__':
#     testMSPortfolio()
    constructModelData()